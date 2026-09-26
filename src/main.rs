use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use memmap2::Mmap;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::channel;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const NUM_TEXTURES: u32 = 1000;
const TEX_SIZE: u32 = 512;
const TILES_PER_AXIS: u32 = TEX_SIZE / 16;
const TILES_PER_TEXTURE: u32 = TILES_PER_AXIS * TILES_PER_AXIS;
const TOTAL_TILES: u32 = NUM_TEXTURES * TILES_PER_TEXTURE;
const MAX_RESIDENT_SLOTS: u32 = 1000;
const IO_WORKERS: usize = 4;
const MAX_UPLOADS_PER_FRAME: usize = 64;

const PREFETCH_PROJ_SIZE: f32 = 24.0;
const UNLOAD_PROJ_SIZE: f32 = 16.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompressParams {
    in_w: u32,
    in_h: u32,
    tex_w: u32,
    tex_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    quant_scale: u32,
    num_textures: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderParams {
    tex_w: u32,
    tex_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    quant_scale: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera {
    view_proj: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct TileHeader {
    base_offset: u32,
    count: u32,
    dc: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VTexHeader {
    magic: [u8; 4],
    num_textures: u32,
    tex_w: u32,
    tex_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    dc_grid_offset: u64,
    deltas_offset: u64,
    headers_offset: u64,
    total_file_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneInstance {
    pos: [f32; 3],
    rot: f32,
    size: [f32; 2],
    tex_id: u32,
    _pad: u32,
}

struct DynamicVramAllocator {
    free_indices: Vec<u32>,
    allocated_mask: Vec<bool>,
    capacity: u32,
}

impl DynamicVramAllocator {
    fn new(capacity: u32) -> Self {
        Self {
            free_indices: (0..capacity).rev().collect(),
            allocated_mask: vec![false; capacity as usize],
            capacity,
        }
    }

    fn allocate(&mut self) -> Option<u32> {
        self.free_indices.pop().map(|idx| {
            self.allocated_mask[idx as usize] = true;
            idx
        })
    }

    fn free(&mut self, idx: u32) {
        if self.allocated_mask[idx as usize] {
            self.allocated_mask[idx as usize] = false;
            self.free_indices.push(idx);
        }
    }

    fn active_count(&self) -> usize {
        (self.capacity as usize).saturating_sub(self.free_indices.len())
    }
}

#[derive(Clone)]
struct StreamRequest {
    tex_id: u32,
    vram_idx: u32,
    ticket: u64,
    priority: f32,
}

impl PartialEq for StreamRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for StreamRequest {}

impl PartialOrd for StreamRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StreamRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

struct StreamResponse {
    tex_id: u32,
    vram_idx: u32,
    ticket: u64,
    headers_data: Vec<u8>,
}

struct WorkQueue {
    heap: BinaryHeap<StreamRequest>,
    cancelled_tickets: HashSet<u64>,
}

struct SharedQueue {
    queue: Mutex<WorkQueue>,
    cvar: Condvar,
}

struct SlotRecord {
    vram_idx: u32,
    is_resident: bool,
    is_loading: bool,
    ticket: u64,
    last_proj_pixels: f32,
}

fn generate_procedural_fallback(w: u32, h: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            let check = ((x / 32) + (y / 32)) % 2 == 0;
            let grad = (x % 256) as u8;
            if check {
                bytes[idx] = 220;
                bytes[idx + 1] = grad;
                bytes[idx + 2] = 50;
                bytes[idx + 3] = 255;
            } else {
                bytes[idx] = 30;
                bytes[idx + 1] = 60;
                bytes[idx + 2] = grad;
                bytes[idx + 3] = 255;
            }
        }
    }
    bytes
}

fn build_perspective_matrix(fov_rad: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_rad * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) * nf, -1.0,
        0.0, 0.0, (2.0 * far * near) * nf, 0.0,
    ]
}

fn build_view_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> ([f32; 16], [f32; 3], [f32; 3], [f32; 3]) {
    let f = {
        let dx = target[0] - eye[0];
        let dy = target[1] - eye[1];
        let dz = target[2] - eye[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        [dx / len, dy / len, dz / len]
    };
    let s = {
        let cx = f[1] * up[2] - f[2] * up[1];
        let cy = f[2] * up[0] - f[0] * up[2];
        let cz = f[0] * up[1] - f[1] * up[0];
        let len = (cx * cx + cy * cy + cz * cz).sqrt();
        [cx / len, cy / len, cz / len]
    };
    let u = [
        s[1] * f[2] - s[2] * f[1],
        s[2] * f[0] - s[0] * f[2],
        s[0] * f[1] - s[1] * f[0],
    ];
    let tx = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    let ty = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    let tz = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];

    (
        [
            s[0], u[0], -f[0], 0.0,
            s[1], u[1], -f[1], 0.0,
            s[2], u[2], -f[2], 0.0,
            tx, ty, tz, 1.0,
        ],
        f,
        s,
        u,
    )
}

fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn main() {
    let quant_scale: u32 = 4;
    let initial_w: u32 = 1920;
    let initial_h: u32 = 1080;

    let input_path = Path::new("input.png");
    let (in_w, in_h, raw_rgba) = if input_path.exists() {
        let img = image::open(input_path).unwrap().to_rgba8();
        (img.width(), img.height(), img.into_raw())
    } else {
        let (w, h) = (4096, 4096);
        let bytes = generate_procedural_fallback(w, h);
        RgbaImage::from_raw(w, h, bytes.clone()).unwrap().save("input.png").unwrap();
        (w, h, bytes)
    };

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("CWST v0.1 by Grisha Sladky")
            .with_inner_size(winit::dpi::PhysicalSize::new(initial_w, initial_h))
            .build(&event_loop)
            .unwrap(),
    );

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let has_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let req_features = if has_timestamps {
        wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    };

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: req_features,
            required_limits: adapter.limits(),
        },
        None,
    ))
    .unwrap();

    let timestamp_period = queue.get_timestamp_period();
    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];

    let present_mode = surface_caps
        .present_modes
        .iter()
        .copied()
        .find(|&m| m == wgpu::PresentMode::AutoNoVsync || m == wgpu::PresentMode::Immediate)
        .unwrap_or(wgpu::PresentMode::Fifo);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let create_depth = |dev: &wgpu::Device, w: u32, h: u32| {
        let tex = dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    };

    let mut depth_res = create_depth(&device, config.width, config.height);

    let (query_set, query_resolve_buf, query_readback_buf) = if has_timestamps {
        (
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp Query Set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })),
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Query Resolve Buffer"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })),
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Query Readback Buffer"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
        )
    } else {
        (None, None, None)
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Unified LoD Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let storage_entry_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let compress_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compress Batch Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry_rw(2),
            storage_entry_rw(3),
            storage_entry_rw(4),
        ],
    });

    let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Empty Layout"),
        entries: &[],
    });

    let render_storage_ro = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Render Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            render_storage_ro(2, wgpu::ShaderStages::FRAGMENT),
            render_storage_ro(3, wgpu::ShaderStages::FRAGMENT),
            render_storage_ro(4, wgpu::ShaderStages::VERTEX),
            render_storage_ro(5, wgpu::ShaderStages::FRAGMENT),
            render_storage_ro(6, wgpu::ShaderStages::FRAGMENT),
            render_storage_ro(7, wgpu::ShaderStages::FRAGMENT),
        ],
    });

    let compress_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compress Pipeline Layout"),
        bind_group_layouts: &[&compress_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&empty_layout, &render_layout],
        push_constant_ranges: &[],
    });

    let compress_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compress Pipeline"),
        layout: Some(&compress_pipeline_layout),
        module: &shader,
        entry_point: "compress_batch",
        compilation_options: Default::default(),
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Instanced Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let vtex_path = Path::new("world_textures.vtex");
    if !vtex_path.exists() {
        let comp_params = CompressParams {
            in_w,
            in_h,
            tex_w: TEX_SIZE,
            tex_h: TEX_SIZE,
            tiles_x: TILES_PER_AXIS,
            tiles_y: TILES_PER_AXIS,
            quant_scale,
            num_textures: NUM_TEXTURES,
        };

        let comp_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compress Params"),
            contents: bytemuck::bytes_of(&comp_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let raw_input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raw Input Pixels"),
            contents: &raw_rgba,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let total_headers_size = (TOTAL_TILES as usize * std::mem::size_of::<TileHeader>()) as u64;
        let all_tile_headers_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Raw Headers Buffer"),
            size: total_headers_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let max_deltas_capacity = (NUM_TEXTURES as usize * 120_000 * 4) as u64;
        let all_global_deltas_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Raw Deltas Buffer"),
            size: max_deltas_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Delta Counter"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let counter_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Read Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compress_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batch Compress Bind Group"),
            layout: &compress_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: comp_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: raw_input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: all_tile_headers_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: all_global_deltas_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: counter_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&compress_pipeline);
            cpass.set_bind_group(0, &compress_bind_group, &[]);
            cpass.dispatch_workgroups(TILES_PER_AXIS, TILES_PER_AXIS, NUM_TEXTURES);
        }
        enc.copy_buffer_to_buffer(&counter_buf, 0, &counter_read_buf, 0, 4);
        queue.submit(Some(enc.finish()));

        let slice_cnt = counter_read_buf.slice(..);
        let (tx_c, rx_c) = channel();
        slice_cnt.map_async(wgpu::MapMode::Read, move |r| tx_c.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx_c.recv().unwrap().unwrap();
        let total_pass1_deltas = {
            let view = slice_cnt.get_mapped_range();
            let val: &[u32] = bytemuck::cast_slice(&view);
            val[0]
        };
        counter_read_buf.unmap();

        let pass1_deltas_size = (total_pass1_deltas * 4) as u64;

        let staging_headers = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Headers"),
            size: total_headers_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_deltas = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Deltas"),
            size: pass1_deltas_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc_read = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc_read.copy_buffer_to_buffer(&all_tile_headers_buf, 0, &staging_headers, 0, total_headers_size);
        if pass1_deltas_size > 0 {
            enc_read.copy_buffer_to_buffer(&all_global_deltas_buf, 0, &staging_deltas, 0, pass1_deltas_size);
        }
        queue.submit(Some(enc_read.finish()));

        let slice_h = staging_headers.slice(..);
        let (tx_h, rx_h) = channel();
        slice_h.map_async(wgpu::MapMode::Read, move |r| tx_h.send(r).unwrap());

        let slice_d = staging_deltas.slice(..);
        let (tx_d, rx_d) = channel();
        slice_d.map_async(wgpu::MapMode::Read, move |r| tx_d.send(r).unwrap());

        device.poll(wgpu::Maintain::Wait);
        rx_h.recv().unwrap().unwrap();
        rx_d.recv().unwrap().unwrap();

        let raw_headers: Vec<TileHeader> = bytemuck::cast_slice(&slice_h.get_mapped_range()).to_vec();
        let raw_deltas: Vec<u32> = bytemuck::cast_slice(&slice_d.get_mapped_range()).to_vec();
        staging_headers.unmap();
        staging_deltas.unmap();

        let mut dc_grid = Vec::with_capacity(TOTAL_TILES as usize);
        for h in &raw_headers {
            dc_grid.push(h.dc);
        }

        let mut global_dag_dict: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut global_dag_deltas: Vec<u32> = Vec::new();
        let mut all_factored_headers: Vec<TileHeader> = Vec::with_capacity(raw_headers.len());

        for chunk in raw_headers.chunks(TILES_PER_TEXTURE as usize) {
            let mut tex_headers = chunk.to_vec();
            for h in &mut tex_headers {
                let cnt = h.count as usize;
                if cnt == 0 {
                    h.base_offset = 0;
                } else {
                    let st = h.base_offset as usize;
                    let mut sorted_seq = raw_deltas[st..st + cnt].to_vec();
                    sorted_seq.sort_by(|a, b| {
                        let pow_a = ((a >> 24) & 0xFF).trailing_zeros();
                        let pow_b = ((b >> 24) & 0xFF).trailing_zeros();
                        pow_b.cmp(&pow_a)
                    });

                    let shared_off = *global_dag_dict.entry(sorted_seq.clone()).or_insert_with(|| {
                        let off = global_dag_deltas.len() as u32;
                        global_dag_deltas.extend_from_slice(&sorted_seq);
                        off
                    });
                    h.base_offset = shared_off;
                }
            }
            all_factored_headers.extend_from_slice(&tex_headers);
        }

        let header_size = std::mem::size_of::<VTexHeader>() as u64;
        let dc_grid_size = (TOTAL_TILES as usize * 4) as u64;
        let deltas_size = (global_dag_deltas.len() * 4) as u64;
        let headers_size = (all_factored_headers.len() * std::mem::size_of::<TileHeader>()) as u64;

        let dc_grid_offset = header_size;
        let deltas_offset = dc_grid_offset + dc_grid_size;
        let headers_offset = deltas_offset + deltas_size;
        let total_file_size = headers_offset + headers_size;

        let mut f = File::create(vtex_path).unwrap();
        let vtex_header = VTexHeader {
            magic: *b"VTEX",
            num_textures: NUM_TEXTURES,
            tex_w: TEX_SIZE,
            tex_h: TEX_SIZE,
            tiles_x: TILES_PER_AXIS,
            tiles_y: TILES_PER_AXIS,
            dc_grid_offset,
            deltas_offset,
            headers_offset,
            total_file_size,
        };

        f.write_all(bytemuck::bytes_of(&vtex_header)).unwrap();
        f.write_all(bytemuck::cast_slice(&dc_grid)).unwrap();
        f.write_all(bytemuck::cast_slice(&global_dag_deltas)).unwrap();
        f.write_all(bytemuck::cast_slice(&all_factored_headers)).unwrap();
    }

    let file = File::open(vtex_path).unwrap();
    let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });

    let vtex_header: &VTexHeader = bytemuck::from_bytes(&mmap[0..std::mem::size_of::<VTexHeader>()]);
    let dc_start = vtex_header.dc_grid_offset as usize;
    let dc_end = dc_start + (TOTAL_TILES as usize * 4);
    let dc_grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Global DC Grid Buffer"),
        contents: &mmap[dc_start..dc_end],
        usage: wgpu::BufferUsages::STORAGE,
    });

    let deltas_start = vtex_header.deltas_offset as usize;
    let deltas_end = vtex_header.headers_offset as usize;
    let all_global_deltas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Global Shared Deltas Buffer"),
        contents: &mmap[deltas_start..deltas_end],
        usage: wgpu::BufferUsages::STORAGE,
    });

    let headers_file_start = vtex_header.headers_offset as usize;
    let slot_headers_size = (MAX_RESIDENT_SLOTS as u64) * (TILES_PER_TEXTURE as u64) * 16;
    let slot_headers_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dynamic Slot Headers Buffer"),
        size: slot_headers_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let tex_lookup = vec![0xFFFFFFFFu32; NUM_TEXTURES as usize];
    let mut current_tex_lookup = tex_lookup.clone();
    let slot_lookup_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tex Resident Lookup Buffer"),
        contents: bytemuck::cast_slice(&current_tex_lookup),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let mut tex_alphas = vec![0.0f32; NUM_TEXTURES as usize];
    let tex_alphas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tex Continuous Alphas Buffer"),
        contents: bytemuck::cast_slice(&tex_alphas),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let mut instances = Vec::with_capacity(NUM_TEXTURES as usize);
    for i in 0..NUM_TEXTURES {
        let fi = i as f32;
        let ring = (i / 100) as f32;
        let angle = fi * 2.39996;
        let radius = 4.0 + (fi % 12.0) * 2.2 + ring * 2.5;
        let z = 3.0 + (fi / 10.0) * 1.35;
        let x = angle.cos() * radius;
        let y = (angle.sin() * radius * 0.6) - 1.0;
        let rot = (fi * 0.17) % 6.283;
        instances.push(PlaneInstance {
            pos: [x, y, z],
            rot,
            size: [2.6, 2.6],
            tex_id: i,
            _pad: 0,
        });
    }

    let instances_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("1000 Plane Instances Buffer"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let render_params = RenderParams {
        tex_w: TEX_SIZE,
        tex_h: TEX_SIZE,
        tiles_x: TILES_PER_AXIS,
        tiles_y: TILES_PER_AXIS,
        quant_scale,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let render_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Render Params Buffer"),
        contents: bytemuck::bytes_of(&render_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut camera = Camera {
        view_proj: [0.0; 16],
    };

    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::bytes_of(&camera),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Empty Bind Group"),
        layout: &empty_layout,
        entries: &[],
    });

    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Render 1000 Bind Group"),
        layout: &render_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: render_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: slot_headers_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: all_global_deltas_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: instances_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: slot_lookup_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: dc_grid_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: tex_alphas_buf.as_entire_binding() },
        ],
    });

    let shared_queue = Arc::new(SharedQueue {
        queue: Mutex::new(WorkQueue {
            heap: BinaryHeap::new(),
            cancelled_tickets: HashSet::new(),
        }),
        cvar: Condvar::new(),
    });

    let (res_tx, res_rx) = channel::<StreamResponse>();

    for _ in 0..IO_WORKERS {
        let sq = Arc::clone(&shared_queue);
        let tx = res_tx.clone();
        let mmap_ref = Arc::clone(&mmap);
        let header_chunk_bytes = (TILES_PER_TEXTURE as usize) * 16;

        std::thread::spawn(move || {
            loop {
                let req = {
                    let mut lock = sq.queue.lock().unwrap();
                    loop {
                        if let Some(top) = lock.heap.pop() {
                            if lock.cancelled_tickets.remove(&top.ticket) {
                                continue;
                            }
                            break top;
                        }
                        lock = sq.cvar.wait(lock).unwrap();
                    }
                };

                let st = headers_file_start + (req.tex_id as usize) * header_chunk_bytes;
                let end = st + header_chunk_bytes;
                let headers_data = mmap_ref[st..end].to_vec();

                let _ = tx.send(StreamResponse {
                    tex_id: req.tex_id,
                    vram_idx: req.vram_idx,
                    ticket: req.ticket,
                    headers_data,
                });
            }
        });
    }

    let mut vram_allocator = DynamicVramAllocator::new(MAX_RESIDENT_SLOTS);
    let mut slots_info: Vec<Option<SlotRecord>> = (0..NUM_TEXTURES).map(|_| None).collect();
    let mut global_ticket: u64 = 0;

    let mut sys = sysinfo::System::new_all();
    let current_pid = sysinfo::get_current_pid().ok();
    let current_os_pid = std::process::id();
    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f32;
    let nvml = nvml_wrapper::Nvml::init().ok();

    let app_start = Instant::now();
    let mut last_stats_instant = Instant::now();
    let mut last_frame_instant = Instant::now();

    let mut app_cpu_pct: f32 = 0.0;
    let mut app_ram_mb: f64 = 0.0;
    let mut gpu_core_pct: u32 = 0;
    let mut app_vram_mb: f64 = 0.0;

    let fov_rad = 1.15f32;
    let tan_half_fov = (fov_rad * 0.5).tan();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(&device, &config);
                        depth_res = create_depth(&device, config.width, config.height);
                    }
                    WindowEvent::RedrawRequested => {
                        let dt = last_frame_instant.elapsed().as_secs_f32().min(0.05);
                        last_frame_instant = Instant::now();

                        let t = app_start.elapsed().as_secs_f32();
                        let cam_z = (t * 7.5) % 135.0;
                        let cam_x = (t * 0.45).sin() * 5.5;
                        let cam_y = (t * 0.35).cos() * 2.2;

                        let eye = [cam_x, cam_y, cam_z];
                        let target = [cam_x * 0.4, cam_y * 0.3, cam_z + 20.0];
                        let up = [0.0, 1.0, 0.0];

                        let aspect = config.width as f32 / config.height as f32;
                        let proj = build_perspective_matrix(fov_rad, aspect, 0.8, 180.0);
                        let (view, cam_f, cam_s, cam_u) = build_view_matrix(eye, target, up);
                        camera.view_proj = mat4_mul(proj, view);
                        queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

                        let mut lookup_changed = false;
                        let mut uploads_this_frame = 0;

                        while let Ok(resp) = res_rx.try_recv() {
                            let tid = resp.tex_id as usize;
                            if let Some(rec) = slots_info[tid].as_mut() {
                                if rec.ticket == resp.ticket && rec.vram_idx == resp.vram_idx {
                                    let h_offset = (resp.vram_idx as u64) * (TILES_PER_TEXTURE as u64) * 16;
                                    queue.write_buffer(&slot_headers_buf, h_offset, &resp.headers_data);

                                    rec.is_resident = true;
                                    rec.is_loading = false;
                                    current_tex_lookup[tid] = resp.vram_idx;
                                    lookup_changed = true;

                                    uploads_this_frame += 1;
                                    if uploads_this_frame >= MAX_UPLOADS_PER_FRAME {
                                        break;
                                    }
                                }
                            }
                        }

                        let screen_h = config.height as f32;
                        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(NUM_TEXTURES as usize);

                        for (idx, inst) in instances.iter().enumerate() {
                            let rel_x = inst.pos[0] - eye[0];
                            let rel_y = inst.pos[1] - eye[1];
                            let rel_z = inst.pos[2] - eye[2];

                            let z_cam = rel_x * cam_f[0] + rel_y * cam_f[1] + rel_z * cam_f[2];
                            let x_cam = rel_x * cam_s[0] + rel_y * cam_s[1] + rel_z * cam_s[2];
                            let y_cam = rel_x * cam_u[0] + rel_y * cam_u[1] + rel_z * cam_u[2];

                            let radius = (inst.size[0].max(inst.size[1])) * 0.75;
                            let half_h = z_cam * tan_half_fov;
                            let half_w = half_h * aspect;

                            let in_frustum = z_cam >= 0.8
                                && z_cam <= 180.0
                                && (x_cam.abs() - radius) <= half_w
                                && (y_cam.abs() - radius) <= half_h;

                            let proj_pixels = if in_frustum {
                                let max_side = inst.size[0].max(inst.size[1]);
                                (max_side * screen_h) / (2.0 * z_cam * tan_half_fov)
                            } else {
                                0.0
                            };

                            if let Some(rec) = slots_info[idx].as_mut() {
                                rec.last_proj_pixels = proj_pixels;
                            }

                            if proj_pixels >= PREFETCH_PROJ_SIZE {
                                candidates.push((idx as u32, proj_pixels));
                            } else if let Some(rec) = slots_info[idx].take() {
                                if proj_pixels < UNLOAD_PROJ_SIZE {
                                    shared_queue.queue.lock().unwrap().cancelled_tickets.insert(rec.ticket);
                                    vram_allocator.free(rec.vram_idx);
                                    current_tex_lookup[idx] = 0xFFFFFFFF;
                                    tex_alphas[idx] = 0.0;
                                    lookup_changed = true;
                                } else {
                                    slots_info[idx] = Some(rec);
                                }
                            }
                        }

                        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

                        let mut new_requests = Vec::new();
                        let mut retained_candidates = HashSet::new();

                        for &(tid, proj_pixels) in candidates.iter().take(MAX_RESIDENT_SLOTS as usize) {
                            let t_idx = tid as usize;
                            retained_candidates.insert(tid);

                            if let Some(rec) = slots_info[t_idx].as_mut() {
                                if !rec.is_resident && !rec.is_loading {
                                    global_ticket += 1;
                                    rec.ticket = global_ticket;
                                    rec.is_loading = true;
                                    new_requests.push(StreamRequest {
                                        tex_id: tid,
                                        vram_idx: rec.vram_idx,
                                        ticket: global_ticket,
                                        priority: proj_pixels,
                                    });
                                }
                            } else if let Some(vram_idx) = vram_allocator.allocate() {
                                global_ticket += 1;
                                slots_info[t_idx] = Some(SlotRecord {
                                    vram_idx,
                                    is_resident: false,
                                    is_loading: true,
                                    ticket: global_ticket,
                                    last_proj_pixels: proj_pixels,
                                });
                                new_requests.push(StreamRequest {
                                    tex_id: tid,
                                    vram_idx,
                                    ticket: global_ticket,
                                    priority: proj_pixels,
                                });
                            } else {
                                let mut min_proj = proj_pixels;
                                let mut victim_tid = None;

                                for (i, slot_opt) in slots_info.iter().enumerate() {
                                    if let Some(rec) = slot_opt {
                                        if !retained_candidates.contains(&(i as u32)) && rec.last_proj_pixels < min_proj {
                                            min_proj = rec.last_proj_pixels;
                                            victim_tid = Some(i);
                                        }
                                    }
                                }

                                if let Some(v_tid) = victim_tid {
                                    let v_rec = slots_info[v_tid].take().unwrap();
                                    shared_queue.queue.lock().unwrap().cancelled_tickets.insert(v_rec.ticket);
                                    current_tex_lookup[v_tid] = 0xFFFFFFFF;
                                    tex_alphas[v_tid] = 0.0;
                                    lookup_changed = true;

                                    global_ticket += 1;
                                    slots_info[t_idx] = Some(SlotRecord {
                                        vram_idx: v_rec.vram_idx,
                                        is_resident: false,
                                        is_loading: true,
                                        ticket: global_ticket,
                                        last_proj_pixels: proj_pixels,
                                    });
                                    new_requests.push(StreamRequest {
                                        tex_id: tid,
                                        vram_idx: v_rec.vram_idx,
                                        ticket: global_ticket,
                                        priority: proj_pixels,
                                    });
                                }
                            }
                        }

                        if !new_requests.is_empty() {
                            let mut q = shared_queue.queue.lock().unwrap();
                            for req in new_requests {
                                q.heap.push(req);
                            }
                            shared_queue.cvar.notify_all();
                        }

                        if lookup_changed {
                            queue.write_buffer(&slot_lookup_buf, 0, bytemuck::cast_slice(&current_tex_lookup));
                        }

                        let alpha_step = dt / 0.15;
                        for i in 0..NUM_TEXTURES as usize {
                            if let Some(rec) = slots_info[i].as_ref() {
                                if rec.is_resident {
                                    tex_alphas[i] = (tex_alphas[i] + alpha_step).min(1.0);
                                } else {
                                    tex_alphas[i] = (tex_alphas[i] - alpha_step).max(0.0);
                                }
                            } else {
                                tex_alphas[i] = (tex_alphas[i] - alpha_step).max(0.0);
                            }
                        }
                        queue.write_buffer(&tex_alphas_buf, 0, bytemuck::cast_slice(&tex_alphas));

                        let output = surface.get_current_texture().unwrap();
                        let view_target = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        {
                            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("1000 Planes Render Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view_target,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.03,
                                            g: 0.03,
                                            b: 0.05,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                    view: &depth_res.1,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                }),
                                timestamp_writes: query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                                    query_set: qs,
                                    beginning_of_pass_write_index: Some(0),
                                    end_of_pass_write_index: Some(1),
                                }),
                                occlusion_query_set: None,
                            });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, &empty_bind_group, &[]);
                            rpass.set_bind_group(1, &render_bind_group, &[]);
                            rpass.draw(0..6, 0..NUM_TEXTURES);
                        }

                        if let (Some(qs), Some(res_b), Some(rb_b)) =
                            (&query_set, &query_resolve_buf, &query_readback_buf)
                        {
                            enc.resolve_query_set(qs, 0..2, res_b, 0);
                            enc.copy_buffer_to_buffer(res_b, 0, rb_b, 0, 16);
                        }

                        let t_render_start = Instant::now();
                        queue.submit(Some(enc.finish()));
                        output.present();
                        let frame_ms = t_render_start.elapsed().as_secs_f64() * 1000.0;

                        let mut gpu_kernel_us = 0.0;
                        if let Some(rb_b) = &query_readback_buf {
                            let slice = rb_b.slice(..);
                            let (tx_ts, rx_ts) = channel();
                            slice.map_async(wgpu::MapMode::Read, move |res| tx_ts.send(res).unwrap());
                            device.poll(wgpu::Maintain::Wait);
                            rx_ts.recv().unwrap().unwrap();
                            let data = slice.get_mapped_range();
                            let ts: &[u64] = bytemuck::cast_slice(&data);
                            if ts[1] >= ts[0] {
                                let diff_ticks = ts[1] - ts[0];
                                gpu_kernel_us = (diff_ticks as f64) * (timestamp_period as f64) / 1000.0;
                            }
                            drop(data);
                            rb_b.unmap();
                        }

                        if last_stats_instant.elapsed().as_millis() >= 120 {
                            last_stats_instant = Instant::now();

                            if let Some(pid) = current_pid {
                                sys.refresh_processes_specifics(
                                    sysinfo::ProcessRefreshKind::new().with_cpu().with_memory(),
                                );
                                if let Some(proc_) = sys.process(pid) {
                                    app_ram_mb = proc_.memory() as f64 / 1_048_576.0;
                                    app_cpu_pct = proc_.cpu_usage() / num_cpus;
                                }
                            }

                            if let Some(n) = &nvml {
                                if let Ok(dev) = n.device_by_index(0) {
                                    if let Ok(util) = dev.utilization_rates() {
                                        gpu_core_pct = util.gpu;
                                    }
                                    if let Ok(procs) = dev.running_graphics_processes() {
                                        for p in procs {
                                            if p.pid == current_os_pid {
                                                if let nvml_wrapper::enums::device::UsedGpuMemory::Used(bytes) = p.used_gpu_memory {
                                                    app_vram_mb = bytes as f64 / 1_048_576.0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            let active_vram_count = vram_allocator.active_count();
                            let resident_kb = (active_vram_count * (TILES_PER_TEXTURE as usize) * 16) as f64 / 1024.0;
                            let file_mb = std::fs::metadata(vtex_path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                            let title = format!(
                                "GPU Pass: {:.1}µs | FPS: {:.0} | Slots: {}/{} ({:.0} KB) | File: {:.1} MB | GPU: {}%  | CPU: {:.1}% | RAM: {:.1}MB",
                                gpu_kernel_us,
                                1000.0 / frame_ms.max(0.001),
                                active_vram_count,
                                MAX_RESIDENT_SLOTS,
                                resident_kb,
                                file_mb,
                                gpu_core_pct,
                                //app_vram_mb,
                                app_cpu_pct,
                                app_ram_mb
                            );
                            window.set_title(&title);
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}