#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec3 vertexColor;
out vec2 TexCoord;
out vec3 v_pos; // Gửi vị trí không gian View sang Fragment Shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Vị trí vật thể trong không gian thế giới (World Space)
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Vị trí vật thể trong không gian camera (View Space) để tính Depth Map chuẩn
    v_pos = vec3(view * model * vec4(aPos, 1.0));
    
    // Chuẩn hóa pháp tuyến
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    
    vertexColor = aColor;
    TexCoord = aTexCoord;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}