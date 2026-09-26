struct CompressParams {
    in_w: u32,
    in_h: u32,
    tex_w: u32,
    tex_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    quant_scale: u32,
    num_textures: u32,
};

struct RenderParams {
    tex_w: u32,
    tex_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    quant_scale: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Camera {
    view_proj: mat4x4<f32>,
};

struct TileHeader {
    base_offset: u32,
    count: u32,
    dc: u32,
    _pad: u32,
};

struct PlaneInstance {
    pos: vec3<f32>,
    rot: f32,
    size: vec2<f32>,
    tex_id: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> comp_params: CompressParams;
@group(0) @binding(1) var<storage, read> raw_input_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> all_tile_headers_rw: array<TileHeader>;
@group(0) @binding(3) var<storage, read_write> all_global_deltas_rw: array<u32>;
@group(0) @binding(4) var<storage, read_write> delta_counter: atomic<u32>;

@group(1) @binding(0) var<uniform> render_params: RenderParams;
@group(1) @binding(1) var<uniform> camera: Camera;
@group(1) @binding(2) var<storage, read> slot_headers: array<TileHeader>;
@group(1) @binding(3) var<storage, read> all_global_deltas: array<u32>;
@group(1) @binding(4) var<storage, read> instances: array<PlaneInstance>;
@group(1) @binding(5) var<storage, read> slot_lookup: array<u32>;
@group(1) @binding(6) var<storage, read> dc_grid: array<u32>;
@group(1) @binding(7) var<storage, read> tex_alphas: array<f32>;

var<workgroup> lds_pixels: array<vec4<i32>, 256>;
var<workgroup> lds_deltas: array<vec4<i32>, 256>;
var<workgroup> scan_buf: array<u32, 256>;
var<workgroup> tile_base_offset: u32;

fn get_local_coords(m: u32) -> vec2<u32> {
    let x = ((m >> 0u) & 1u) | (((m >> 2u) & 1u) << 1u) | (((m >> 4u) & 1u) << 2u) | (((m >> 6u) & 1u) << 3u);
    let y = ((m >> 1u) & 1u) | (((m >> 3u) & 1u) << 1u) | (((m >> 5u) & 1u) << 2u) | (((m >> 7u) & 1u) << 3u);
    return vec2<u32>(x, y);
}

fn encode_morton(x: u32, y: u32) -> u32 {
    var m = 0u;
    m |= (x & 1u) << 0u;
    m |= (y & 1u) << 1u;
    m |= (x & 2u) << 1u;
    m |= (y & 2u) << 2u;
    m |= (x & 4u) << 2u;
    m |= (y & 4u) << 3u;
    m |= (x & 8u) << 3u;
    m |= (y & 8u) << 4u;
    return m;
}

fn rgb_to_ycocg(c: vec4<i32>) -> vec4<i32> {
    let r = c.x;
    let g = c.y;
    let b = c.z;
    let a = c.w;
    let co = r - b;
    let t = b + (co >> 1u);
    let cg = g - t;
    let y = t + (cg >> 1u);
    return vec4<i32>(y, co, cg, a);
}

fn ycocg_to_rgb(ycc: vec4<i32>) -> vec4<i32> {
    let y = ycc.x;
    let co = ycc.y;
    let cg = ycc.z;
    let a = ycc.w;
    let t = y - (cg >> 1u);
    let g = t + cg;
    let b = t - (co >> 1u);
    let r = b + co;
    return vec4<i32>(r, g, b, a);
}

fn unpack_rgba_to_ycocg(c: u32) -> vec4<i32> {
    let rgba = vec4<i32>(
        i32(c & 0xFFu),
        i32((c >> 8u) & 0xFFu),
        i32((c >> 16u) & 0xFFu),
        i32((c >> 24u) & 0xFFu)
    );
    return rgb_to_ycocg(rgba);
}

fn pack_rgba(c: vec4<i32>) -> u32 {
    let clamped = clamp(c, vec4<i32>(0), vec4<i32>(255));
    return u32(clamped.x) | (u32(clamped.y) << 8u) | (u32(clamped.z) << 16u) | (u32(clamped.w) << 24u);
}

fn get_quant_step_pow(pow: u32, q_scale: u32) -> vec4<i32> {
    let q = i32(q_scale);
    switch (pow) {
        case 0u: {
            return vec4<i32>(q * 2, q * 3, q * 3, 1);
        }
        case 1u: {
            return vec4<i32>(q + (q >> 1u), q * 2, q * 2, 1);
        }
        case 2u: {
            return vec4<i32>(q, q + (q >> 1u), q + (q >> 1u), 1);
        }
        case 3u: {
            let y = max(1, q >> 1u);
            return vec4<i32>(y, q, q, 1);
        }
        default: {
            return vec4<i32>(1, 1, 1, 1);
        }
    }
}

fn get_quant_step(step: u32, q_scale: u32) -> vec4<i32> {
    let pow = countTrailingZeros(step);
    return get_quant_step_pow(pow, q_scale);
}

fn pack_sparse_delta(local_idx: u32, qd: vec4<i32>) -> u32 {
    let y = u32(clamp(qd.x, -64, 63) + 64) & 0x7Fu;
    let co = u32(clamp(qd.y, -32, 31) + 32) & 0x3Fu;
    let cg = u32(clamp(qd.z, -32, 31) + 32) & 0x3Fu;
    let a = u32(clamp(qd.w, -16, 15) + 16) & 0x1Fu;
    let payload = y | (co << 7u) | (cg << 13u) | (a << 19u);
    return (local_idx << 24u) | payload;
}

fn unpack_sparse_delta(entry: u32, Q: vec4<i32>) -> vec4<i32> {
    let payload = entry & 0x00FFFFFFu;
    let y = i32(payload & 0x7Fu) - 64;
    let co = i32((payload >> 7u) & 0x3Fu) - 32;
    let cg = i32((payload >> 13u) & 0x3Fu) - 32;
    let a = i32((payload >> 19u) & 0x1Fu) - 16;
    return vec4<i32>(y, co, cg, a) * Q;
}

@compute @workgroup_size(256)
fn compress_batch(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tex_id = wg_id.z;
    let tiles_per_texture = comp_params.tiles_x * comp_params.tiles_y;
    let local_tile_id = wg_id.y * comp_params.tiles_x + wg_id.x;
    let global_tile_id = tex_id * tiles_per_texture + local_tile_id;

    let tid = local_id.x;
    let local_coord = get_local_coords(tid);
    let gx = wg_id.x * 16u + local_coord.x;
    let gy = wg_id.y * 16u + local_coord.y;

    var u_norm = f32(gx) / f32(comp_params.tex_w);
    var v_norm = f32(gy) / f32(comp_params.tex_h);

    if ((tex_id & 1u) != 0u) {
        u_norm = 1.0 - u_norm;
    }
    if ((tex_id & 2u) != 0u) {
        v_norm = 1.0 - v_norm;
    }

    let src_x = min(u32(u_norm * f32(comp_params.in_w)), comp_params.in_w - 1u);
    let src_y = min(u32(v_norm * f32(comp_params.in_h)), comp_params.in_h - 1u);

    let raw_pixel = raw_input_pixels[src_y * comp_params.in_w + src_x];
    var ycc = unpack_rgba_to_ycocg(raw_pixel);

    let angle = f32(tex_id) * 0.12566;
    let cos_a = i32(cos(angle) * 32.0);
    let sin_a = i32(sin(angle) * 32.0);
    let co_rot = (ycc.y * cos_a - ycc.z * sin_a) >> 5u;
    let cg_rot = (ycc.y * sin_a + ycc.z * cos_a) >> 5u;
    ycc.y = clamp(co_rot, -255, 255);
    ycc.z = clamp(cg_rot, -255, 255);

    lds_pixels[tid] = ycc;
    lds_deltas[tid] = vec4<i32>(0);
    workgroupBarrier();

    for (var step = 1u; step < 256u; step *= 2u) {
        let pair_step = step * 2u;
        if ((tid % pair_step) == 0u) {
            let idx_a = tid;
            let idx_b = tid + step;
            let a = lds_pixels[idx_a];
            let b = lds_pixels[idx_b];
            let diff = a - b;

            let Q = get_quant_step(step, comp_params.quant_scale);
            let half_Q = Q >> vec4<u32>(1u);
            let s = sign(diff);
            let abs_d = abs(diff);

            let q = select(vec4<i32>(0), s * ((abs_d + half_Q) / Q), abs_d >= Q);
            let recon_diff = q * Q;
            let avg = b + (recon_diff >> vec4<u32>(1u));

            lds_pixels[idx_a] = avg;
            lds_deltas[idx_b] = q;
        }
        workgroupBarrier();
    }

    var is_nonzero = 0u;
    if (tid > 0u) {
        let q = lds_deltas[tid];
        if (q.x != 0 || q.y != 0 || q.z != 0 || q.w != 0) {
            is_nonzero = 1u;
        }
    }

    scan_buf[tid] = is_nonzero;
    workgroupBarrier();

    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var temp = 0u;
        if (tid >= offset) {
            temp = scan_buf[tid - offset];
        }
        workgroupBarrier();
        scan_buf[tid] += temp;
        workgroupBarrier();
    }

    let total_nnz = scan_buf[255];
    let local_slot = select(0u, scan_buf[tid - 1u], tid > 0u);

    if (tid == 0u) {
        tile_base_offset = atomicAdd(&delta_counter, total_nnz);
        all_tile_headers_rw[global_tile_id].base_offset = tile_base_offset;
        all_tile_headers_rw[global_tile_id].count = total_nnz;
        all_tile_headers_rw[global_tile_id].dc = pack_rgba(ycocg_to_rgb(lds_pixels[0]));
        all_tile_headers_rw[global_tile_id]._pad = 0u;
    }
    workgroupBarrier();

    if (is_nonzero == 1u) {
        all_global_deltas_rw[tile_base_offset + local_slot] = pack_sparse_delta(tid, lds_deltas[tid]);
    }
}

fn sample_dc_grid(tex_id: u32, uv: vec2<f32>) -> vec4<f32> {
    let uv_clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let grid_f = vec2<f32>(f32(render_params.tiles_x), f32(render_params.tiles_y));
    let coord = uv_clamped * grid_f - vec2<f32>(0.5);
    let base_coord = floor(coord);
    let f = coord - base_coord;

    let x0 = clamp(i32(base_coord.x), 0, i32(render_params.tiles_x) - 1);
    let y0 = clamp(i32(base_coord.y), 0, i32(render_params.tiles_y) - 1);
    let x1 = min(x0 + 1, i32(render_params.tiles_x) - 1);
    let y1 = min(y0 + 1, i32(render_params.tiles_y) - 1);

    let base_tex = tex_id * (render_params.tiles_x * render_params.tiles_y);
    let c00 = unpack_rgba_to_ycocg(dc_grid[base_tex + u32(y0) * render_params.tiles_x + u32(x0)]);
    let c10 = unpack_rgba_to_ycocg(dc_grid[base_tex + u32(y0) * render_params.tiles_x + u32(x1)]);
    let c01 = unpack_rgba_to_ycocg(dc_grid[base_tex + u32(y1) * render_params.tiles_x + u32(x0)]);
    let c11 = unpack_rgba_to_ycocg(dc_grid[base_tex + u32(y1) * render_params.tiles_x + u32(x1)]);

    let top = mix(vec4<f32>(c00), vec4<f32>(c10), f.x);
    let bot = mix(vec4<f32>(c01), vec4<f32>(c11), f.x);
    let ycc = mix(top, bot, f.y);

    let rgb = ycocg_to_rgb(vec4<i32>(round(ycc)));
    return clamp(vec4<f32>(rgb) / 255.0, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn eval_sample_tap(slot_id: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let pixel_coords = uv * vec2<f32>(f32(render_params.tex_w), f32(render_params.tex_h));
    let px = clamp(u32(pixel_coords.x), 0u, render_params.tex_w - 1u);
    let py = clamp(u32(pixel_coords.y), 0u, render_params.tex_h - 1u);

    let tile_x = px / 16u;
    let tile_y = py / 16u;
    let tiles_per_texture = render_params.tiles_x * render_params.tiles_y;
    let slot_tile_id = slot_id * tiles_per_texture + (tile_y * render_params.tiles_x + tile_x);

    let header = slot_headers[slot_tile_id];
    var accum = vec4<f32>(unpack_rgba_to_ycocg(header.dc));

    let count = header.count;
    if (count > 0u && lod < 7.0) {
        let morton_id = encode_morton(px % 16u, py % 16u);
        let base = header.base_offset;
        let min_allowed_step_pow = u32(clamp(floor(lod), 0.0, 7.0));

        for (var i = 0u; i < count; i = i + 1u) {
            let entry = all_global_deltas[base + i];
            let local_idx = (entry >> 24u) & 0xFFu;
            let step_pow = countTrailingZeros(local_idx);

            if (step_pow < min_allowed_step_pow) {
                break;
            }

            let step = 1u << step_pow;
            let expected_idx = (morton_id & ~(2u * step - 1u)) + step;

            if (local_idx == expected_idx) {
                let Q = get_quant_step_pow(step_pow, render_params.quant_scale);
                let diff = vec4<f32>(unpack_sparse_delta(entry, Q));
                let dir = select(0.0, 1.0, (morton_id & step) != 0u);
                let half_d = floor(diff * 0.5);
                let term = select(diff - half_d, -half_d, dir == 1.0);

                let weight = clamp(1.0 - max(0.0, lod - f32(step_pow)), 0.0, 1.0);
                accum += term * weight;
            }
        }
    }

    let rgb = ycocg_to_rgb(vec4<i32>(round(accum)));
    return clamp(vec4<f32>(rgb) / 255.0, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn sample_bin(tex_id: u32, uv: vec2<f32>) -> vec4<f32> {
    let slot_id = slot_lookup[tex_id];
    let alpha = tex_alphas[tex_id];

    if (slot_id == 0xFFFFFFFFu || alpha <= 0.0) {
        return sample_dc_grid(tex_id, uv);
    }

    let ddx = dpdx(uv) * vec2<f32>(f32(render_params.tex_w), f32(render_params.tex_h));
    let ddy = dpdy(uv) * vec2<f32>(f32(render_params.tex_w), f32(render_params.tex_h));
    let footprint = max(length(ddx), length(ddy));

    if (footprint >= 128.0) {
        let dc_tile = eval_sample_tap(slot_id, uv, 7.0);
        if (alpha >= 1.0) {
            return dc_tile;
        }
        return mix(sample_dc_grid(tex_id, uv), dc_tile, alpha);
    }

    let lod = log2(max(footprint, 1.0));
    var wavelet_color: vec4<f32>;

    if (footprint < 1.5) {
        let tex_size = vec2<f32>(f32(render_params.tex_w), f32(render_params.tex_h));
        let tex_coord = uv * tex_size - 0.5;
        let base_coord = floor(tex_coord);
        let f = tex_coord - base_coord;

        let inv_size = 1.0 / tex_size;
        let uv00 = (base_coord + 0.5) * inv_size;
        let uv10 = uv00 + vec2<f32>(inv_size.x, 0.0);
        let uv01 = uv00 + vec2<f32>(0.0, inv_size.y);
        let uv11 = uv00 + inv_size;

        let c00 = eval_sample_tap(slot_id, uv00, lod);
        let c10 = eval_sample_tap(slot_id, uv10, lod);
        let c01 = eval_sample_tap(slot_id, uv01, lod);
        let c11 = eval_sample_tap(slot_id, uv11, lod);

        let top = mix(c00, c10, f.x);
        let bot = mix(c01, c11, f.x);
        wavelet_color = mix(top, bot, f.y);
    } else {
        wavelet_color = eval_sample_tap(slot_id, uv, lod);
    }

    if (alpha >= 1.0) {
        return wavelet_color;
    }

    return mix(sample_dc_grid(tex_id, uv), wavelet_color, alpha);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) tex_id: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VertexOutput {
    let inst = instances[iid];

    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5)
    );

    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );

    let v = quad_verts[vid] * inst.size;
    let cos_r = cos(inst.rot);
    let sin_r = sin(inst.rot);
    let local_x = v.x * cos_r - v.y * sin_r;
    let local_y = v.x * sin_r + v.y * cos_r;

    let world_pos = vec4<f32>(inst.pos.x + local_x, inst.pos.y + local_y, inst.pos.z, 1.0);

    var out: VertexOutput;
    out.position = camera.view_proj * world_pos;
    out.uv = uvs[vid];
    out.tex_id = inst.tex_id;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return sample_bin(in.tex_id, in.uv);
}