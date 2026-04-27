"""
OpenGL Shaders for rendering 3D Meshes in ModernGL.
"""

SCENE_VERTEX_SHADER = """
#version 330 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * world;
}
"""

SCENE_FRAGMENT_SHADER = """
#version 330 core

uniform vec3 u_light_pos;
uniform vec3 u_cam_pos;
uniform vec3 u_color;
uniform sampler2D u_texture;
uniform bool u_use_texture;

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

layout(location = 0) out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    // Use a fixed directional light (e.g. sunlight) instead of a point light
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(N, light_dir), 0.0);

    vec3 base_color = u_color;
    if (u_use_texture) {
        base_color = texture(u_texture, v_uv).rgb;
    }

    // Increase ambient so shadows aren't pitch black
    vec3 ambient = 0.5 * base_color;
    // Lower diffuse slightly so it doesn't overexpose when combined with higher ambient
    vec3 diffuse = 0.6 * diff * base_color;

    // Specular highlights
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 H = normalize(light_dir + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = spec * vec3(0.1); // subtle specular

    frag_color = vec4(ambient + diffuse + specular, 1.0);
}
"""
