#version 330 core

out vec4 fColor;

uniform vec3 color;

void main(){
	fColor.rgb = color;
}