#version 120

uniform vec2  uCenter, uCenterHi, uCenterLo;
uniform float uZoom;
uniform ivec2 uResolution;
uniform int   uMaxIter;
uniform float uBrightness;   // try 1.15
uniform float uGamma;        // try 1.80
uniform float uContrast;     // try 1.25

// ---- Cosine palette ----
vec3 cosPalette(float t, vec3 a, vec3 b, vec3 c, vec3 d){
    return a + b * cos(6.2831853 * (c * t + d));
}

// Neutral monochrome palette (subtle cool bias, no pink)
vec3 palNeutral(float t){
    // identical 'c' keeps channels nearly equal -> greys
    vec3 col = cosPalette(t,
        vec3(0.34),          // base grey
        vec3(0.62),          // amplitude
        vec3(0.82),          // frequency
        vec3(0.08,0.10,0.12) // very slight RGB phase offset (cool tint)
    );
    return clamp(col, 0.0, 1.0);
}

vec3 colorize(vec2 c){
    vec2  z = vec2(0.0);
    float r2 = 0.0;
    float trap = 1e20;
    int   it = 0;

    for (int i=0;i<20000;++i){
        if (i >= uMaxIter) break;
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        r2 = dot(z,z);
        trap = min(trap, abs(z.x)+abs(z.y));
        if (r2 > 4.0){ it = i; break; }
        if (i == uMaxIter-1) it = uMaxIter;
    }

    // In-set: dark graphite
    if (it >= uMaxIter) return vec3(0.03, 0.035, 0.04);

    // Smooth iterations
    float mu = float(it) + 1.0 - log2(log2(sqrt(r2)));
    float t  = mu / float(uMaxIter);

    // More shades (no candy bands)
    float bands = 3.2;                 // 2.5–5.0
    float tt = fract(t * bands);

    vec3 col = palNeutral(tt);

    // Orbit-trap highlight (very gentle)
    float trapW = smoothstep(0.0, 0.35, 1.0 - clamp(trap, 0.0, 2.0));
    col = mix(col, vec3(0.95), trapW * 0.25);

    // Edge bloom a touch
    float edge = clamp(1.0 - tt*tt*2.0, 0.0, 1.0);
    col = mix(col, vec3(1.0), edge * 0.06);

    // Heavy desaturation → mostly greys (but not flat)
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), 0.85); // 0.85 = 85% towards grey

    // Tone ops
    col = clamp(col, 0.0, 1.0);
    col = (col - 0.5) * max(uContrast, 0.0) + 0.5;
    col *= uBrightness;
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(uGamma, 0.0001)));
    return clamp(col, 0.0, 1.0);
}

void main(){
    vec2 res    = vec2(uResolution);
    vec2 px     = 1.0 / res;
    vec2 split  = uCenterHi + uCenterLo;
    vec2 center = (dot(split, split) > 0.0) ? split : uCenter;

    vec2 uv = gl_FragCoord.xy * px * 2.0 - 1.0;
    float aspect = res.x / res.y;
    uv.x *= aspect;

    vec3 col;

    if (uZoom < 4e-4){
        vec2 offs[4];
        offs[0]=vec2(-0.25,-0.25); offs[1]=vec2( 0.25,-0.25);
        offs[2]=vec2(-0.25, 0.25); offs[3]=vec2( 0.25, 0.25);
        vec3 acc = vec3(0.0);
        for (int i=0;i<4;++i){
            vec2 uv2 = (gl_FragCoord.xy + offs[i]) * px * 2.0 - 1.0;
            uv2.x *= aspect;
            acc += colorize(center + uv2 * uZoom);
        }
        col = acc * 0.25;
    }else{
        col = colorize(center + uv * uZoom);
    }

    gl_FragColor = vec4(col, 1.0);
}
