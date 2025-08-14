#version 120

uniform vec2  uCenter;
uniform float uZoom;
uniform ivec2 uResolution;
uniform int   uMaxIter;

vec3 palette(float t){
    return 0.5 + 0.5*cos(6.2831853*(vec3(0.0,0.33,0.67)+t));
}

void main(){
    vec2 res = vec2(uResolution);
    vec2 uv  = (gl_FragCoord.xy / res) * 2.0 - 1.0;
    uv.x *= res.x / res.y;

    vec2 c = uCenter + uv * uZoom;
    vec2 z = vec2(0.0);
    int i;
    for(i=0;i<uMaxIter && dot(z,z) < 4.0; ++i){
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
    }
    if(i==uMaxIter){
        gl_FragColor = vec4(0.0,0.0,0.0,1.0); // interior black
    }else{
        float mu = float(i) - log2(log2(dot(z,z))) + 1.0;
        float t = mu / float(uMaxIter);
        vec3 col = palette(t);
        gl_FragColor = vec4(vec3(1.0) - col,1.0);
    }
}
