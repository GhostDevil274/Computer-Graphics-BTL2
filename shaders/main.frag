#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 vertexColor;
in vec2 TexCoord;
in vec3 v_pos;

uniform int render_mode;
uniform vec3 flat_color;
uniform sampler2D tex_diffuse;
uniform vec3 viewPos;
uniform vec3 bg_color;
uniform int is_depth_map;
uniform int is_mask_map;
uniform bool light1_on;
uniform bool light2_on;
uniform bool light3_on;

// Đưa về hằng số (const) để tối ưu bộ nhớ GPU
const vec3 sunLightDir = normalize(vec3(0.5, 1.0, 0.5));
const vec3 sunLightColor = vec3(1.4, 1.4, 1.4);
const vec3 pointLight1Pos = vec3(1.5, 3.0, 2.0); 
const vec3 pointLight1Color = vec3(4.0, 2.0, 0.0); 
const vec3 pointLight2Pos = vec3(-1.5, 3.0, 2.0);
const vec3 pointLight2Color = vec3(0.0, 2.0, 4.0);

vec3 calcDirLight(vec3 norm, vec3 viewDir, vec3 baseCol) {
    float diff = max(dot(norm, sunLightDir), 0.0);
    vec3 reflectDir = reflect(-sunLightDir, norm);
    // SỬA LỖI MAC: Đổi 32 thành 32.0 (Bắt buộc dùng số thực float)
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    return (diff * sunLightColor * baseCol) + (0.5 * spec * sunLightColor);
}

vec3 calcPointLight(vec3 lightPos, vec3 lightCol, vec3 norm, vec3 viewDir, vec3 baseCol) {
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    float dist = length(lightPos - FragPos);
    float atten = 1.0 / (1.0 + 0.045 * dist + 0.0075 * (dist * dist));
    return ((diff * lightCol * baseCol) + (0.5 * spec * lightCol)) * atten;
}

void main() {
    // 1. CHẾ ĐỘ MASK: Trả về màu bệt duy nhất (Không tính ánh sáng)
    if (is_mask_map == 1) {
        FragColor = vec4(flat_color, 1.0);
        return;
    }

    // 2. CHẾ ĐỘ DEPTH MAP: Tính khoảng cách vật lý
    if (is_depth_map == 1) {
        float near = 0.1;
        float far = 50.0; // Thầy có hỏi thì nhớ nhắc tới dải khoảng cách này
        float depth = length(v_pos);
        float scaled = clamp((depth - near) / (far - near), 0.0, 1.0);
        FragColor = vec4(vec3(scaled), 1.0);
        return;
    }

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 baseCol;

    // 3. TÌM MÀU CƠ BẢN DỰA VÀO RENDER MODE
    // 0: Flat, 1: Vertex, 2: Phong(Flat), 3: Texture, 4: Combo(Phong+Texture)
    if (render_mode == 0 || render_mode == 2) {
        baseCol = flat_color;
    } else if (render_mode == 1) {
        baseCol = vertexColor;
    } else {
        baseCol = texture(tex_diffuse, TexCoord).rgb;
    }

    // 4. ÁP DỤNG ÁNH SÁNG
    if (render_mode == 0 || render_mode == 1 || render_mode == 3) {
        FragColor = vec4(baseCol, 1.0);
    } else {
        vec3 lighting = 0.2 * baseCol; // Ambient
        if (light1_on) lighting += calcDirLight(norm, viewDir, baseCol);
        if (light2_on) lighting += calcPointLight(pointLight1Pos, pointLight1Color, norm, viewDir, baseCol);
        if (light3_on) lighting += calcPointLight(pointLight2Pos, pointLight2Color, norm, viewDir, baseCol);
        FragColor = vec4(lighting, 1.0);
    }
}