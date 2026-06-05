// interference.frag
#version 120
// Inputs
uniform vec2  uResolution;      // pixelstorlek (t.ex. 1920,1080)
uniform float uTime;            // tid i sekunder
uniform float uFreq;            // våg-frekvens i pixlar^-1 (t.ex. 0.05)
uniform float uPhase1;          // extra fas till källa 1
uniform float uPhase2;          // extra fas till källa 2
uniform vec2  uOrbitCenter;     // centrum för cirkelbanorna (i pixlar)
uniform float uOrbitRadius;     // radie för banorna (pixlar)
uniform float uSpeed1;          // vinkelhastighet rad/s
uniform float uSpeed2;          // vinkelhastighet rad/s
uniform float uGamma;           // gamma justering, ~1.6..2.2

// Hjälp: enkel tonemapping
float toGamma(float x, float g) {
    return pow(clamp(x, 0.0, 1.0), 1.0 / g);
}

void main() {
    vec2 p = gl_FragCoord.xy; // pixelpos i fönsterkoordinater

    // Rörliga källor i cirkelbanor kring uOrbitCenter
    float a1 = uSpeed1 * uTime;
    float a2 = uSpeed2 * uTime;

    vec2 c1 = uOrbitCenter + vec2(cos(a1), sin(a1)) * uOrbitRadius;
    vec2 c2 = uOrbitCenter + vec2(cos(a2), sin(a2)) * uOrbitRadius;

    // Avstånd
    float d1 = length(p - c1);
    float d2 = length(p - c2);

    // Två sinusvågor + valfria fasskift
    float w = sin(d1 * uFreq + uPhase1) + sin(d2 * uFreq + uPhase2);

    // Normalisera ~0..1 och gamma-korrigera
    float L = (w + 2.0) * 0.25;        // (-2..+2) -> (0..1)
    L = toGamma(L, uGamma);

    gl_FragColor = vec4(vec3(L), 1.0);
}
