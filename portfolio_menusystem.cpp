// OpenSans-Regular.ttf eller Roboto-VariableFont_wdth,wght.ttf
// logo.png
// backgroundmusic.wav
// musik1.wav, musik2.wav, musik3.wav, musik4.wav

/*
   _____    _____   _    _   _____    ______  __      __
  / ____|  / ____| | |  | | |  __ \  |  ____| \ \    / /
 | (___   | |      | |__| | | |  | | | |__     \ \  / /
  \___ \  | |      |  __  | | |  | | |  __|     \ \/ /
  ____) | | |____  | |  | | | |__| | | |____     \  /
 |_____/   \_____| |_|  |_| |_____/  |______|     \/
(c) 2025 by Mats Schyllander, made with help from ChatGPT, SDL2 and Open-GL community.
*/

#ifndef SDL_MAIN_HANDLED
#define SDL_MAIN_HANDLED
#endif

// --- Includes ---
#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <SDL_mixer.h>
#include <GL/glew.h>
#include <SDL_opengl.h>
#ifdef _WIN32
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#endif
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <fstream> 
#include <sstream>
#include <tuple>
#include <random>


SDL_Window* gWindow = nullptr;
SDL_GLContext gGL = nullptr;

// --- C64 10 PRINT: forward declarations (måste synas före första anropet) ---
void initC64Window(int screenW, int screenH);
void startC64Window();
void updateC64Window(float dt);
bool c64WindowIsDone();
void c64RequestFly();
void renderC64Window(int screenW, int screenH);

// Ritar en SDL_Texture som ett overlay i GL-pixelkoordinater (0..screenW, 0..screenH)
void glDrawSDLTextureOverlay(SDL_Texture* tex, int x, int y, int w, int h, int screenW, int screenH) {
    if (!tex) return;

    // Gör ortho (compat), ingen depth/cull
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, screenW, screenH, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_TEXTURE_2D);

    float tw, th;
    if (SDL_GL_BindTexture(tex, &tw, &th) != 0) return; // renderer måste vara OpenGL-backend

    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f((float)x, (float)y);
    glTexCoord2f(tw, 0); glVertex2f((float)(x + w), (float)y);
    glTexCoord2f(tw, th); glVertex2f((float)(x + w), (float)(y + h));
    glTexCoord2f(0, th); glVertex2f((float)x, (float)(y + h));
    glEnd();

    SDL_GL_UnbindTexture(tex);
    glDisable(GL_TEXTURE_2D);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// --- C64 shaders (globala strängar) ---
static const char* c64_vs = R"GLSL(
#version 330 core
layout (location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

// === C64 10 PRINT — vertex shader (panel + bg i en pass) ===
static const char* c64_fs = R"GLSL(
// C64 10 PRINT — fragment shader (panel + decrunch bg + gated reveal)
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uRand;        // GL_R8, size == uGrid
uniform ivec2  uGrid;           // gridW, gridH
uniform int    uRevealCount;    // hur många celler visade
uniform float  uThickness;      // linjetjocklek (cell-UV)
uniform vec3   uInk;            // linjefärg (nära vit)
uniform vec3   uBG;             // panel bg (C64-blå)
uniform vec3   uEdge;           // panelens kantfärg
uniform vec4   uPanelRect;      // {cx,cy,w,h} i [0..1], center + size
uniform float  uTime;           // sekunder
uniform float  uDecomp;         // 0..1, dekomprimeringsprogress

// SDF: avstånd till linjesegment
float sdSegment(vec2 p, vec2 a, vec2 b){
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/max(dot(ba,ba),1e-6), 0.0, 1.0);
    return length(pa - ba*h);
}

// Litet brus
float hash21(vec2 p){
    p = fract(p*vec2(123.34, 456.21));
    p += dot(p, p + 34.345);
    return fract(p.x*p.y);
}

// Decrunch-bakgrund med scanbars/snow/progressbar
vec3 decrunchBG(vec2 uv, float t, float prog){
    float bars = smoothstep(0.0, 0.2, 0.5 + 0.5 * sin(uv.y*120.0 + t*40.0));
    float snow = step(0.965, hash21(uv*vec2(640.0, 360.0) + t*60.0));
    float band = smoothstep(0.45,0.47, abs(uv.y-0.7));
    float scan = 0.25*bars + 0.10*snow + 0.08*band;

    float barY = 0.06, barH = 0.012;
    float x = uv.x, y = uv.y;
    float frameOK = step(barY, y) * step(y, barY+barH) * step(0.12, x) * step(x, 0.88);
    float fill    = step(0.12, x) * step(x, mix(0.12, 0.88, prog)) * step(barY, y) * step(y, barY+barH);

    vec3 base     = vec3(0.02, 0.03, 0.05) + scan;
    vec3 frameCol = vec3(0.10, 1.00, 0.70);
    vec3 fillCol  = vec3(0.20, 0.90, 0.40);

    float edgeX = step(0.119, x) * step(x, 0.881);
    float edgeY1 = step(barY, y) * step(y, barY + 0.002);
    float edgeY2 = step(barY + barH - 0.002, y) * step(y, barY + barH);

    vec3 progBar = base + frameCol * (edgeX * (edgeY1 + edgeY2)) + fillCol * fill * 0.9;
    // Om vi inte ligger ovanför progressbaren, lämna bara base+scan
    return mix(base, progBar, frameOK);
}

void main(){
    vec2 uv = vUV;

    // Panel-rect
    vec2 c = uPanelRect.xy;
    vec2 s = uPanelRect.zw;
    vec2 minP = c - 0.5*s;
    vec2 maxP = c + 0.5*s;

    bool inPanel = all(greaterThanEqual(uv, minP)) && all(lessThanEqual(uv, maxP));

    // Utanför panelen: kör decrunch tills klar, SEN transparent så SDL kan synas
    if (!inPanel) {
        if (uDecomp < 1.0) {
            FragColor = vec4(decrunchBG(uv, uTime, uDecomp), 1.0);
        } else {
            FragColor = vec4(0.0); // transparent
        }
        return;
    }

    // Panel-UV [0..1]
    vec2 puv = clamp((uv - minP) / max(s, vec2(1e-6)), 0.0, 1.0);

    // Tunn bevel/ram
    float pad = 0.015;
    float inner = step(pad, puv.x) * step(pad, puv.y) * step(pad, 1.0-puv.x) * step(pad, 1.0-puv.y);
    vec3  base = mix(uEdge, uBG, 0.95);
    vec3  panelCol = mix(uEdge * 0.6 + uBG * 0.4, base, inner);

    // Under decrunch: visa bara panelfärg (ingen reveal ännu)
    if (uDecomp < 1.0) {
        FragColor = vec4(panelCol, 1.0);
        return;
    }

    // Grid + reveal i TOP-LEFT-ordning (radvis)
    ivec2 grid  = uGrid;
    vec2  g     = vec2(grid);
    vec2  cellf = floor(puv * g);

    int colIdx      = clamp(int(cellf.x), 0, grid.x - 1);              // 0..W-1, vänster->höger
    int rowFromTop  = clamp(grid.y - 1 - int(cellf.y), 0, grid.y - 1); // topp->botten
    int idx         = rowFromTop * grid.x + colIdx;

    if (idx >= uRevealCount) {
        FragColor = vec4(panelCol, 1.0);
        return;
    }

    // Lokala cell-coords
    ivec2 cell = ivec2(colIdx, clamp(int(cellf.y), 0, grid.y - 1));
    vec2  f    = fract(puv * g);

    // '/' eller '\'
    float pick = texelFetch(uRand, cell, 0).r;

    // Lite fetare C64-slash (justeras med uThickness)
    float d = (pick < 0.5)
        ? sdSegment(f, vec2(0.05,0.05), vec2(0.95,0.95))
        : sdSegment(f, vec2(0.05,0.95), vec2(0.95,0.05));

    // Linjemask (sätt uThickness ~ 0.12 från CPU)
    float lineMask = smoothstep(uThickness, 0.0, d);

    // Subtila scanlines i panelen
    float scan = 1.0 - 0.06 * (0.5 + 0.5 * sin(puv.y * 1200.0));

    // Slutfärg
    vec3 finalCol = mix(panelCol, uInk, lineMask) * scan;
    FragColor = vec4(clamp(finalCol, 0.0, 1.0), 1.0);
}
)GLSL";


static inline void ensureGLContextCurrent() {
    if (SDL_GL_GetCurrentContext() != gGL) {
        SDL_GL_MakeCurrent(gWindow, gGL);
    }
}

// --- Plasma globals (måste finnas före användning) ---
static SDL_Texture* gPlasma = nullptr;
static int gPlasmaW = 25, gPlasmaH = 20;
static SDL_PixelFormat* gARGB = nullptr;   // set i main: gARGB = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);

// Framåtdeklarationer
static void ensurePlasmaTexture(SDL_Renderer* ren);
static void updatePlasmaTexture(SDL_Texture* tex, float time,
float speed, float scrollU, float scrollV);

// FractalZoom state (görs en gång per fil)
static int    f_currentTarget = 0;
static float  f_zoom = 1.0f;
static float  f_phaseTime = 0.f;
static float  f_centerX = -0.75f, f_centerY = 0.0f;
static float  f_startX, f_startY, f_targetX, f_targetY;

constexpr int FRACTAL_NUM_TARGETS = 3;
static const struct { float x, y; } f_targets[FRACTAL_NUM_TARGETS] = {
    { -0.7487729488352518f, 0.1238947751881455f },
    { -0.748771809192f,     0.123894376955f     },
    { -0.74877294883525f,   0.12389477518814f   }
};

enum FractalPhase { FRACTAL_FLY, FRACTAL_HOLD };
static FractalPhase f_phase = FRACTAL_FLY;
enum C64Phase { C64_FILLING, C64_HOLD, C64_FLY, C64_DONE };

GLuint loadShader(const char* vertexPath, const char* fragmentPath) {
    auto loadFile = [](const char* path) -> std::string {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Could not open shader file: " << path << std::endl;
            return "";
        }
        return std::string(std::istreambuf_iterator<char>(file), {});
        };

    std::string vertexCode = loadFile(vertexPath);
    std::string fragmentCode = loadFile(fragmentPath);
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    GLuint vertex, fragment;
    GLint success;
    char infoLog[512];

    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertex, 512, NULL, infoLog);
        std::cerr << "Vertex Shader Compilation Failed:\n" << infoLog << std::endl;
    }

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragment, 512, NULL, infoLog);
        std::cerr << "Fragment Shader Compilation Failed:\n" << infoLog << std::endl;
    }

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertex);
    glAttachShader(shaderProgram, fragment);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader Program Linking Failed:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return shaderProgram;
}

// --- IMPLEMENTATION ---
static SDL_Texture* getDotMask(SDL_Renderer* ren) {
    static SDL_Texture* mask = nullptr;
    if (mask) return mask;

    const int W = 3, H = 3;
    mask = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, W, H);
    if (!mask) return nullptr;

    SDL_SetTextureBlendMode(mask, SDL_BLENDMODE_MOD);
    SDL_SetTextureScaleMode(mask, SDL_ScaleModeNearest);

    void* pixels = nullptr; int pitch = 0;
    if (SDL_LockTexture(mask, nullptr, &pixels, &pitch) == 0) {
        // Använd ett EGET format – ingen gARGB behövs
        SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);

        for (int y = 0; y < H; ++y) {
            Uint32* row = reinterpret_cast<Uint32*>(
                static_cast<Uint8*>(pixels) + y * pitch);
            for (int x = 0; x < W; ++x) {
                bool dot = (x == 1 && y == 1);     // mittpunkten “vit”
                Uint8 v = dot ? 255 : 170;         // runtom lite mörkare
                row[x] = SDL_MapRGBA(fmt, v, v, v, 255);
            }
        }

        SDL_UnlockTexture(mask);
        SDL_FreeFormat(fmt);
    }
    return mask;
}

static SDL_Texture* renderTextCRT(SDL_Renderer* ren, TTF_Font* f,
    const std::string& txt, SDL_Color col) {
    SDL_Surface* s0 = TTF_RenderText_Blended(f, txt.c_str(), col);
    if (!s0) return nullptr;

    SDL_Surface* s = SDL_ConvertSurfaceFormat(s0, SDL_PIXELFORMAT_ARGB8888, 0);
    SDL_FreeSurface(s0);
    if (!s) return nullptr;

    Uint32* p = (Uint32*)s->pixels;
    int w = s->w, h = s->h;
    SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);

    for (int y = 0; y < h; ++y) {
        if ((y & 1) == 0) {               // varannan rad
            for (int x = 0; x < w; ++x) {
                Uint8 r, g, b, a;
                SDL_GetRGBA(p[y * w + x], fmt, &r, &g, &b, &a);
                if (a) {                  // bara glyph-pixlar
                    r = Uint8(r * 0.65f);
                    g = Uint8(g * 0.65f);
                    b = Uint8(b * 0.65f);
                    p[y * w + x] = SDL_MapRGBA(fmt, r, g, b, a);
                }
            }
        }
    }
    SDL_FreeFormat(fmt);

    SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, s);
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
    SDL_FreeSurface(s);
    return tex;
}


static void applyDotMask(SDL_Renderer* ren, const SDL_Rect& dst) {
    SDL_Texture* m = getDotMask(ren);
    if (!m) return;

    SDL_Rect src{ 0,0,3,3 };
    for (int y = dst.y; y < dst.y + dst.h; y += src.h) {
        for (int x = dst.x; x < dst.x + dst.w; x += src.w) {
            SDL_Rect d{
                x, y,
                std::min(src.w, dst.x + dst.w - x),
                std::min(src.h, dst.y + dst.h - y)
            };
            SDL_RenderCopy(ren, m, &src, &d);
        }
    }
}

static void applyScanlines(SDL_Renderer* ren, const SDL_Rect& dst) {
    // tunna, halvtransparenta svarta linjer varannan rad
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 55);
    for (int y = dst.y; y < dst.y + dst.h; y += 2) {
        SDL_RenderDrawLine(ren, dst.x, y, dst.x + dst.w, y);
    }
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}

// 1x2 pattern: dark row + transparent row, tiled vertically
static SDL_Texture* getScanlineTex(SDL_Renderer* ren) {
    static SDL_Texture* tex = nullptr;
    if (tex) return tex;

    tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, 1, 2);
    if (!tex) return nullptr;

    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
    SDL_SetTextureScaleMode(tex, SDL_ScaleModeNearest);

    void* p = nullptr; int pitch = 0;
    if (SDL_LockTexture(tex, nullptr, &p, &pitch) == 0) {
        SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);
        auto px = reinterpret_cast<Uint32*>(p);
        px[0] = SDL_MapRGBA(fmt, 0, 0, 0, 110); // dark line (alpha)
        px[1] = SDL_MapRGBA(fmt, 0, 0, 0, 0); // empty line
        SDL_UnlockTexture(tex);
        SDL_FreeFormat(fmt);
    }
    return tex;
}

static void applyScanlines(SDL_Renderer* ren, const SDL_Rect& dst, int scale = 1) {
    SDL_Texture* sl = getScanlineTex(ren);
    if (!sl) return;

    // tile the 1x2 pattern  -> each copy becomes 2*scale px tall
    SDL_Rect r{ dst.x, dst.y, dst.w, 2 * scale };
    for (int y = dst.y; y < dst.y + dst.h; y += 2 * scale) {
        r.y = y;
        SDL_RenderCopy(ren, sl, nullptr, &r);
    }
}


// Clamp helper for pre-C++17 compilers
template <typename T>
static T clampValue(T val, T minVal, T maxVal) {
    return val < minVal ? minVal : (val > maxVal ? maxVal : val);
}

// --- Constants and globals ---
constexpr float PI = 3.14159265f;
SDL_Texture* backButtonTexture = nullptr;
SDL_Rect     backButtonRect;
SDL_Texture* nextButtonTexture = nullptr;
SDL_Rect     nextButtonRect;
SDL_Texture* quitButtonTexture = nullptr;
SDL_Rect     quitButtonRect;

bool click = false;
static float interPhaseA = 0.f, interPhaseB = 0.f;
// Wireframe fly-out transition
static bool  WF_FlyOut = false;
static float WF_FlyT = 0.f;
static float WF_FlyDur = 1.2f; // längd på fly-out i sekunder
static float MENU_BASE_SCALE = 1.6f;   // större menytext

GLuint mandelbrotShader = 0;
GLuint mandelbrotVAO = 0;
GLuint mandelbrotVBO = 0;


// --- Wireframe settings and helpers ---
static int  WF_LINE_THICKNESS = 1;    // 0 = inga linjer, 1 = tunnast, 2..n = tjockare
static Uint8 WF_LINE_ALPHA = 90;  // grund-alfa (0..255) för linjerna
static inline float easeOutCubic(float x) { x = clampValue(x, 0.f, 1.f); return 1.f - powf(1.f - x, 3.f); }

void initFractalZoom() {
    if (mandelbrotVAO != 0) return;

    // Shader
    mandelbrotShader = loadShader("mandelbrot.vert", "mandelbrot.frag");

    // VAO + VBO
    float quadVertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f
    };

    glGenVertexArrays(1, &mandelbrotVAO);
    glGenBuffers(1, &mandelbrotVBO);

    glBindVertexArray(mandelbrotVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mandelbrotVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

const int NUM_SMALL_STARS = 100;   // fewer stars for lighter tunnel
const int NUM_BIG_STARS = 50;     // adjust big stars accordingly
const int NUM_STATIC_STARS = 120;
const int PONG_MARGIN = 400;
const int PONG_VMARGIN = 150;
const int PONGGAME_MARGIN = 650;
const int PONGGAME_VMARGIN = 280;
static float gLastDt = 1.0f / 60.0f;  // used by the wrapper pong game

const float FRACTAL_ZOOM_SPEED = 0.01f; // speed multiplier for fractal zoom

// Plasma polygon settings
const float SHAPE_DURATION = 10.0f;
float currentView;

// --- Fireworks ---
struct FireworkParticle {
    float x, y, vx, vy, life;
    SDL_Color color;
};

std::vector<FireworkParticle> fireworks;  // riktig definition (inte extern)
float nextBurst = 5.f;                   // riktig definition (inte extern)


struct StaticStar { int x, y; Uint8 baseColor; float alpha, alphaSpeed; int direction; };
struct Star {
    float x, y, z, prevX, prevY, speed;
    Star() { reset(); prevX = screenX(); prevY = screenY(); }
    void reset();
    void update(float mult = 1.f);
    float screenX() const;
    float screenY() const;
};
struct Vec3 { float x, y, z; Vec3() :x(0), y(0), z(0) {} Vec3(float X, float Y, float Z) :x(X), y(Y), z(Z) {} };

static inline float wrapf(float a, float m) { a = fmodf(a, m); if (a < 0) a += m; return a; }

// --- forwards used below ---
Vec3 rotateY(const Vec3& v, float angle);
Vec3 rotateX(const Vec3& v, float angle);

// drawThickLine is used in several renderers before its definition
static void drawThickLine(SDL_Renderer* ren, int x1, int y1, int x2, int y2, int t);

// Radien till en regelbunden k-hörning (circumradius = 1) i riktning theta.
// Formel: r(θ) = cos(π/k) / cos( (θ mod 2π/k) - π/k )VIEW_WIREFRAME_CUBE
static inline float regularPolygonRadius(int k, float theta) {
    const float period = 2.0f * PI / k;
    const float h = PI / k;
    const float ang = wrapf(theta, period);
    const float denom = cosf(ang - h);
    // denom blir aldrig 0 för k>=3; liten clamp för numerisk trygghet:
    return cosf(h) / fmaxf(denom, 1e-4f);
}

// Bygg morf-ringen mellan två polygoner k0→k1 (samples = upplösning)
static void buildMorphRing(int k0, int k1, float t, int samples, float rad, float half,
    float ax, float ay,
    std::vector<Vec3>& top, std::vector<Vec3>& bot) {
    top.resize(samples); bot.resize(samples);
    for (int i = 0; i < samples; ++i) {
        float th = (2.f * PI * i) / samples;

        float r0 = regularPolygonRadius(k0, th);
        float r1 = regularPolygonRadius(k1, th);
        float r = rad * ((1.f - t) * r0 + t * r1);

        Vec3 T{ r * cosf(th), r * sinf(th),  half };
        Vec3 B{ r * cosf(th), r * sinf(th), -half };

        T = rotateX(rotateY(T, ay), ax);
        B = rotateX(rotateY(B, ay), ax);

        top[i] = T; bot[i] = B;
    }
}

enum AppState { STATE_MENU, STATE_PORTFOLIO, STATE_ABOUT };
enum PortfolioSubState : int {
    VIEW_WIREFRAME_CUBE = 0,
    VIEW_PLASMA,
    VIEW_FRACTAL_ZOOM,    
    VIEW_INTERFERENCE,
    VIEW_SNAKE_GAME,
    VIEW_PONG_GAME,
    VIEW_C64_10PRINT
};


void initStaticStars();
void renderStaticStars(SDL_Renderer*);
void updateStars(float deltaTime);
void renderStars(SDL_Renderer* ren, std::vector<Star>& stars, SDL_Color col);
void initMenu();
void renderMenu(float deltaTime, int mouseX, int mouseY, bool mouseClick);
void initPortfolio();
void renderPlasmaPolygon(float time);
Vec3 rotateY(const Vec3&, float angle);
Vec3 rotateX(const Vec3&, float angle);
SDL_Point projectPoint(const Vec3&, int w, int h, float fov, float dist);

void renderWireframeCube(SDL_Renderer*, float angleX, float angleY, SDL_Color);

void renderSolidCubeGL(float angleX, float angleY, float time);
void renderFilledPrismGL(int sides, float ax, float ay, float time);
void renderCylinderGL(float angle);
void renderDotPrismGL(int sides, float ax, float ay, float time);
void drawFilledCircle(float x, float y, float radius, int segments, SDL_Color color);
static void drawGreyCube();
static void drawSolidPrism(int sides);
void renderWireframePrism(SDL_Renderer*, int sides, float ax, float ay, SDL_Color);
SDL_Color rainbowColor(float t);
SDL_Color plasmaColor(int x, int y, float time);

void renderPlasma(SDL_Renderer*, float time);
void renderFractalZoom(SDL_Renderer*, float deltaTime);
void initInterference();
void updateInterference(float dt);
void renderInterference(SDL_Renderer*);
void initSnakeGame();
void updateSnakeGame(float dt);
void renderSnakeGame(SDL_Renderer* ren);
void initPongGame();
void updatePongGame(float dt);
void renderPongGame(SDL_Renderer* ren, float dt);       // main one
void renderFireworks(SDL_Renderer*, float dt);
void renderLogoWithReflection(SDL_Renderer*, SDL_Texture*, int baseX);
void startExitExplosion(bool returnToMenu = false, bool nextEffect = false, int nextIndex = 0);
void startStarTransition(int newIndex, bool toMenu = false);
static void drawThickLine(SDL_Renderer* ren, int x1, int y1, int x2, int y2, int t);
static void drawSmallFilledCircle(SDL_Renderer* ren, int cx, int cy, int radius);
static void renderPrismSidesWithTexture(SDL_Renderer* ren, SDL_Texture* tex,
    int sides, float ax, float ay,
    const SDL_Rect& panel, Uint8 alpha = 255);

static const SDL_Color kMonoGreenBase  = {120, 220, 120, 255};
static const SDL_Color kMonoGreenHover = {170, 255, 170, 255};
static const SDL_Color kCRTTint        = {30, 80, 30, 255};

bool renderPortfolioEffect(SDL_Renderer*, float deltaTime);
void renderAboutC64Typewriter(SDL_Renderer* ren, float dt);

SDL_Renderer* renderer = nullptr;
SDL_Texture* logoTexture = nullptr;
TTF_Font* menuFont = nullptr;
TTF_Font* PongFont = nullptr;
TTF_Font* consoleFont = nullptr;
TTF_Font* bigFont = nullptr;
TTF_Font* titleFont = nullptr;
TTF_Font* scrollFont = nullptr;
Mix_Chunk* hoverSound = nullptr;
Mix_Chunk* bangSound = nullptr;
Mix_Music* backgroundMusic = nullptr;
Mix_Music* demoMusic = nullptr;
Mix_Music* currentMusic = nullptr;
bool portfolioMusicStarted = false;

std::vector<StaticStar> staticStars;
std::vector<Star>       smallStars, bigStars;
float starSpeedMultiplier = 2.5f;
bool starTransition = false;
bool starTransitionToMenu = false;
float starTransitionTime = 0.f;
const float starTransitionDuration = 1.3f;
int targetEffectIndex = 0;
// camera orientation for starfield
float starYaw = 0.f, starPitch = 0.f;
float startYaw = 0.f, endYaw = 0.f;
float startPitch = 0.f, endPitch = 0.f;
bool backWasHovered = false;
bool nextWasHovered = false;
bool quitWasHovered = false;
float backHoverAnim = 0.f;
float nextHoverAnim = 0.f;
float quitHoverAnim = 0.f;
bool exitExplosion = false; 
const char* c64;
float exitTimer = 0.1f;
bool explosionReturnToMenu = false;
bool explosionAdvanceEffect = false;
int  explosionNextIndex = 0;
float fractalZoomProgress = 0.f;


struct InterferenceCircle { float x, y, r, dx, dy; bool white; };
std::vector<InterferenceCircle> interferenceCircles;



static SDL_Point projectPointPanel(const Vec3& v, const SDL_Rect& panel, float fov = 500.f, float dist = 3.f) {
    float factor = fov / (dist + v.z);
    int x = int(v.x * factor + panel.w / 2);
    int y = int(-v.y * factor + panel.h / 2);
    return { panel.x + x, panel.y + y };
}

static inline bool intersectY(const SDL_Point& p0, const SDL_Point& p1, int y, float& x) {
    if (p0.y == p1.y) return false;
    int ymin = std::min(p0.y, p1.y), ymax = std::max(p0.y, p1.y);
    if (y < ymin || y >= ymax) return false;
    float t = (y - p0.y) / float(p1.y - p0.y);
    x = p0.x + t * (p1.x - p0.x);
    return true;
}



// C64 program struktur

struct C64WindowState {
    // Grid
    int gridW = 42;                // typisk C64-breddkänsla, justera fritt
    int gridH = 25;
    int totalCells = gridW * gridH;
    float cellsPerSec = 700.f;     // hur snabbt “PRINT” sker
    int revealCount = 0;

    // i C64WindowState
    float rotX = 0.f, rotY = 0.f;   // aktuella rotationsvinklar (rad)
    float idleAmp = 0.10f;          // ~6 grader
    float idleSpeed = 0.8f;         // svänghastighet
    float zDepth = 0.f;             // Z-förflyttning (0 till negativt vid fly)
    bool  flyRequested = false;     // sätts när man klickar Next (efter fylld ruta)


    // Timing
    C64Phase phase = C64_FILLING;
    float holdTime = 0.6f;         // kort paus när den är full
    float holdT = 0.f;

    // Motion (flyg iväg)
    float tFly = 0.f;
    float flyDuration = 2.2f;

    // Window appearance / transform
    float winWidth = 0.72f;        // som andel av skärmbredd
    float winAR = 4.f / 3.f;         // fönstrets aspekt, C64-känsla
    float alpha = 0.88f;           // bakgrunds-opacity
    float edgeAlpha = 1.0f;
    float thickness = 0.085f;      // linjetjocklek i cell (0..1)

    // 2D transform (NDC-ish)
    float posX = 0.f;              // i NDC-värld: 0 = center
    float posY = 0.f;
    float scale = 1.0f;            // 1.0 = winWidth används direkt

    // OpenGL
    GLuint prog = 0;
    GLuint vao = 0, vbo = 0;
    GLuint randTex = 0;

    // For cleanup / gating
    bool initialized = false;
} C64;


// === C64 10 PRINT — tiny GL utils ===
static GLuint makeShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, 2047, nullptr, log); fprintf(stderr, "C64 shader err: %s\n", log); }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = makeShader(GL_VERTEX_SHADER, vs);
    GLuint f = makeShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, 2047, nullptr, log); fprintf(stderr, "C64 link err: %s\n", log); }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}


// Fyll en triangel med plasmafärger via enkel scanline
static void fillTrianglePlasma(SDL_Renderer* ren, SDL_Point a, SDL_Point b, SDL_Point c, float time) {
    int minY = std::max(0, std::min({ a.y, b.y, c.y }));
    int maxY = std::min(SCREEN_HEIGHT - 1, std::max({ a.y, b.y, c.y }));
    for (int y = minY; y <= maxY; ++y) {
        float xs[3]; int cnt = 0; float xx;
        if (intersectY(a, b, y, xx)) xs[cnt++] = xx;
        if (intersectY(b, c, y, xx)) xs[cnt++] = xx;
        if (intersectY(c, a, y, xx)) xs[cnt++] = xx;
        if (cnt < 2) continue;
        if (xs[0] > xs[1]) std::swap(xs[0], xs[1]);
        int x0 = (int)std::ceil(xs[0]);
        int x1 = (int)std::floor(xs[1]);
        for (int x = x0; x <= x1; ++x) {
            SDL_Color col = plasmaColor(x, y, time * 6.f);
            SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, 220);
            SDL_RenderDrawPoint(ren, x, y);
        }
    }
}

// Hjälpare: plasmafyll triangel men hårdklipp inom panel-rect
static void fillTrianglePlasmaClipped(SDL_Renderer* ren,
    SDL_Point a, SDL_Point b, SDL_Point c,
    float time, const SDL_Rect& clip)
{
    auto interY = [](const SDL_Point& p0, const SDL_Point& p1, int y, float& x) -> bool {
        if (p0.y == p1.y) return false; // horisontell kant → inget snitt
        int ymin = std::min(p0.y, p1.y), ymax = std::max(p0.y, p1.y);
        if (y < ymin || y >= ymax) return false; // half-open [ymin, ymax)
        float t = (y - p0.y) / float(p1.y - p0.y);
        x = p0.x + t * (p1.x - p0.x);
        return true;
        };

    // Klipp vertikalt mot panel
    const int y0 = std::max(clip.y, std::min({ a.y, b.y, c.y }));
    const int y1 = std::min(clip.y + clip.h - 1, std::max({ a.y, b.y, c.y }));

    if (y0 > y1) return;

    for (int y = y0; y <= y1; ++y) {
        float xs[3]; int cnt = 0; float xx;
        if (interY(a, b, y, xx)) xs[cnt++] = xx;
        if (interY(b, c, y, xx)) xs[cnt++] = xx;
        if (interY(c, a, y, xx)) xs[cnt++] = xx;
        if (cnt < 2) continue;

        if (xs[0] > xs[1]) std::swap(xs[0], xs[1]);
        float xL = xs[0], xR = xs[1];

        int x0 = std::max(clip.x, (int)std::ceil(xL));
        int x1 = std::min(clip.x + clip.w - 1, (int)std::floor(xR));
        if (x0 > x1) continue;

        for (int x = x0; x <= x1; ++x) {
            SDL_Color col = plasmaColor(x, y, time * 6.f);
            SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, 220);
            SDL_RenderDrawPoint(ren, x, y);
        }
    }
}


// Ritar sidytorna på ett prisma (sides) med plasma-textur (fyllda väggar, utan sömmar).
static void renderPrismPlasmaFillSDL(SDL_Renderer* ren,
    int sides, float ax, float ay,
    const SDL_Rect& panel, float time)
{
    if (sides < 3) return;

    // Se till att plasma-texturen finns och är uppdaterad (med full alfa)
    ensurePlasmaTexture(ren);
    SDL_SetTextureBlendMode(gPlasma, SDL_BLENDMODE_NONE);   // texturen själv ska inte blenda
    updatePlasmaTexture(gPlasma, time, 3.8f, 0.22f, 0.12f); // genererar RGBA (A=255 i funktionen)

    // Geometri-konstanter (unika namn för att undvika krockar)
    const float kHalf = 1.0f;
    const float kRad = 1.5f;

    struct Face {
        SDL_Point p0, p1, p2, p3;  // screen space
        float     zavg;            // sorteringsnyckel (större = längre bort)
        float     u0, u1;          // U-koordinater runt manteln
    };

    std::vector<Face> faces; faces.reserve((size_t)sides);

    // Roterad topp & botten (värld → kamera)
    std::vector<Vec3> top(sides), bot(sides);
    for (int i = 0; i < sides; ++i) {
        const float th = 2.f * PI * i / sides;
        Vec3 t{ cosf(th) * kRad, sinf(th) * kRad,  kHalf };
        Vec3 b{ cosf(th) * kRad, sinf(th) * kRad, -kHalf };
        top[i] = rotateX(rotateY(t, ay), ax);
        bot[i] = rotateX(rotateY(b, ay), ax);
    }

    // Bygg mantelytor (quad per sida)
    for (int i = 0; i < sides; ++i) {
        const int j = (i + 1) % sides;
        Face f{};
        f.p0 = projectPointPanel(top[i], panel);  // övre vänster
        f.p1 = projectPointPanel(top[j], panel);  // övre höger
        f.p2 = projectPointPanel(bot[j], panel);  // nedre höger
        f.p3 = projectPointPanel(bot[i], panel);  // nedre vänster
        f.zavg = (top[i].z + top[j].z + bot[j].z + bot[i].z) * 0.25f;
        f.u0 = (float)i / (float)sides;
        f.u1 = (float)j / (float)sides;
        faces.push_back(f);
    }

    // Painter's: längst bort först (för säker semi-transparent komposition om du vill använda alfa)
    std::sort(faces.begin(), faces.end(),
        [](const Face& a, const Face& b) { return a.zavg > b.zavg; });

    // Klipp mot panel och rita
    SDL_RenderSetClipRect(ren, &panel);

    // Vi vill att geometri ska blenda om färgen skulle ha alfa, men texturen har A=255 → inget genomskinligt.
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    for (const Face& f : faces) {
        SDL_Vertex v[6];
        auto setV = [](SDL_Vertex& vv, float x, float y, float u, float tv) {
            vv.position.x = x; vv.position.y = y;
            vv.color = SDL_Color{ 255, 255, 255, 255 }; // multiplicera inte ned färgen
            vv.tex_coord.x = u; vv.tex_coord.y = tv;
            };

        // Två trianglar per sida: (p0,p1,p2) och (p0,p2,p3)
        setV(v[0], (float)f.p0.x, (float)f.p0.y, f.u0, 0.f);
        setV(v[1], (float)f.p1.x, (float)f.p1.y, f.u1, 0.f);
        setV(v[2], (float)f.p2.x, (float)f.p2.y, f.u1, 1.f);

        setV(v[3], (float)f.p0.x, (float)f.p0.y, f.u0, 0.f);
        setV(v[4], (float)f.p2.x, (float)f.p2.y, f.u1, 1.f);
        setV(v[5], (float)f.p3.x, (float)f.p3.y, f.u0, 1.f);

        SDL_RenderGeometry(ren, gPlasma, v, 6, nullptr, 0);
    }

    // Städa upp state
    SDL_RenderSetClipRect(ren, nullptr);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}


void initInterference()
{
    // Nollställ faser så start blir deterministisk
    interPhaseA = 0.f;
    interPhaseB = 0.f;

    interferenceCircles.clear();
    interferenceCircles.reserve(5);

    // Minst två grupper – en vit (ADD-blend i render), en mörk (BLEND)
    InterferenceCircle g1{};
    g1.x = SCREEN_WIDTH * 0.30f;
    g1.y = SCREEN_HEIGHT * 0.50f;
    g1.r = 160.f;
    g1.dx = g1.dy = 0.f;
    g1.white = true;
    interferenceCircles.push_back(g1);

    InterferenceCircle g2{};
    g2.x = SCREEN_WIDTH * 0.70f;
    g2.y = SCREEN_HEIGHT * 0.50f;
    g2.r = 160.f;
    g2.dx = g2.dy = 0.f;
    g2.white = false;
    interferenceCircles.push_back(g2);

    // (valfritt) Några extra för fylligare bild
    for (int i = 0; i < 3; ++i) {
        InterferenceCircle c{};
        c.x = float(rand() % SCREEN_WIDTH);
        c.y = float(rand() % SCREEN_HEIGHT);
        c.r = 120.f + float(rand() % 100);
        c.dx = c.dy = 0.f;          // rörelse sköts i updateInterference via faserna
        c.white = ((i & 1) == 0);   // varannan vit/mörk
        interferenceCircles.push_back(c);
    }
}

int logoWidth = 0, logoHeight = 0;
AppState currentState = STATE_MENU;
PortfolioSubState currentPortfolioSubState = VIEW_WIREFRAME_CUBE;


// Sequence configuration for the portfolio showcase
const PortfolioSubState effectSequence[] = {
    VIEW_WIREFRAME_CUBE,
    VIEW_FRACTAL_ZOOM,
    VIEW_SNAKE_GAME,
    VIEW_PONG_GAME,
    VIEW_INTERFERENCE,
    VIEW_C64_10PRINT,
};
const float effectDurations[] = { 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f };
const int NUM_EFFECTS = sizeof(effectSequence) / sizeof(effectSequence[0]);
int currentEffectIndex = 0;
float effectTimer = 0.f;

float cubeAngleX = 0.f, cubeAngleY = 0.f, elapsedTime = 0.f;

void startBackgroundMusic() {
    if (backgroundMusic) {
        Mix_HaltMusic();
        Mix_PlayMusic(backgroundMusic, -1);
        currentMusic = backgroundMusic;
    }
}

float getMusicTime() {
    if (currentMusic) {
        double p = Mix_GetMusicPosition(currentMusic);
        if (p >= 0.0)
            return static_cast<float>(p);
    }
    return elapsedTime;
}


void startPortfolioEffect(PortfolioSubState st) {
    currentPortfolioSubState = st;
    effectTimer = 0.f;

    switch (st) {

    case VIEW_WIREFRAME_CUBE: {
        effectTimer = 0.f;
        break;
    }
    case VIEW_FRACTAL_ZOOM:
        initFractalZoom();
        // Initiera våra fil-scope-variabler:
        f_startX = f_centerX = f_targets[0].x;
        f_startY = f_centerY = f_targets[0].y;
        f_targetX = f_targets[1].x;
        f_targetY = f_targets[1].y;
        f_currentTarget = 0;
        f_phase = FRACTAL_FLY;
        f_phaseTime = 0.f;
        f_zoom = 1.0f;
        break;

    case VIEW_SNAKE_GAME:
        initSnakeGame();
        starYaw = 0.f;
        starPitch = 0.f;
        break;

    case VIEW_PONG_GAME:
        initPongGame();
        starYaw = 0.f;
        starPitch = 0.f;
        break;

    case VIEW_C64_10PRINT:
        initC64Window(SCREEN_WIDTH, SCREEN_HEIGHT);
		starYaw = 0.f;
        break;

    default:
        break;
    }


    if (!portfolioMusicStarted && demoMusic) {
        Mix_HaltMusic();
        Mix_PlayMusic(demoMusic, -1);
        currentMusic = demoMusic;
        portfolioMusicStarted = true;
    }
}

void startStarTransition(int newIndex, bool toMenu) {
    starTransition = true;
    starTransitionToMenu = toMenu;
    starTransitionTime = 0.f;
    targetEffectIndex = newIndex;
    startYaw = 0.f;
    startPitch = 0.f;
    starYaw = 0.f;
    starPitch = 0.f;
    if (!toMenu) {
        endYaw = ((rand() % 200) / 100.f - 1.f) * 0.8f;
        endPitch = ((rand() % 200) / 100.f - 1.f) * 0.5f;
    }
    else {
        endYaw = 0.f;
        endPitch = 0.f;
    }
    for (auto& s : smallStars) s.reset();
    for (auto& s : bigStars) s.reset();
}

struct MenuItem {
    SDL_Rect rect;
    std::string text;
    SDL_Texture* texture;
    float hoverAnim;
    bool wasHovered;
} menuItems[3];

void Star::reset() {
    x = ((float)rand() / RAND_MAX - 0.5f) * 2.4f;
    y = ((float)rand() / RAND_MAX - 0.5f) * 2.4f;
    z = ((float)rand() / RAND_MAX) * 0.9f + 0.1f;
    speed = 0.0025f + ((float)rand() / RAND_MAX) * 0.0045f;
    prevX = screenX();
    prevY = screenY();
}
void Star::update(float mult) {
    prevX = screenX();
    prevY = screenY();
    z -= speed * mult;
    if (z <= 0.01f || z >= 1.f) reset();
}
static void rotateStar(const Star* s, float& rx, float& ry, float& rz) {
    float x1 = s->x;
    float y1 = s->y * cosf(starPitch) - s->z * sinf(starPitch);
    float z1 = s->y * sinf(starPitch) + s->z * cosf(starPitch);
    float x2 = x1 * cosf(starYaw) + z1 * sinf(starYaw);
    float z2 = -x1 * sinf(starYaw) + z1 * cosf(starYaw);
    rx = x2; ry = y1; rz = z2;
}
float Star::screenX() const {
    float rx, ry, rz; rotateStar(this, rx, ry, rz);
    return (rx / rz) * SCREEN_WIDTH / 2 + SCREEN_WIDTH / 2;
}
float Star::screenY() const {
    float rx, ry, rz; rotateStar(this, rx, ry, rz);
    return (ry / rz) * SCREEN_HEIGHT / 2 + SCREEN_HEIGHT / 2;
}

void initStaticStars() {
    staticStars.clear();
    for (int i = 0; i < NUM_STATIC_STARS; ++i) {
        StaticStar s;
        s.x = rand() % SCREEN_WIDTH;
        s.y = rand() % SCREEN_HEIGHT;
        s.baseColor = 180 + rand() % 76;
        s.alpha = float(rand() % 256);
        s.alphaSpeed = 0.2f + (rand() % 100) / 500.0f;
        s.direction = (rand() % 2 ? 1 : -1);
        staticStars.push_back(s);
    }
}
void renderStaticStars(SDL_Renderer* ren) {
    for (auto& s : staticStars) {
        s.alpha += s.alphaSpeed * s.direction;
        if (s.alpha >= 255) { s.alpha = 255; s.direction = -1; }
        else if (s.alpha <= 50) { s.alpha = 50; s.direction = 1; }
        SDL_SetRenderDrawColor(ren, s.baseColor, s.baseColor, s.baseColor, (Uint8)s.alpha);
        SDL_RenderDrawPoint(ren, s.x, s.y);
        if (rand() % 10 == 0) {
            SDL_RenderDrawPoint(ren, s.x + 1, s.y);
            SDL_RenderDrawPoint(ren, s.x, s.y + 1);
        }
    }
}

void renderStars(SDL_Renderer* ren, std::vector<Star>& stars, SDL_Color col) {
    for (auto& st : stars) {
        st.update(starSpeedMultiplier);
        float sx = st.screenX(), sy = st.screenY();
        float px = st.prevX, py = st.prevY;
        if (sx < 0 || sx >= SCREEN_WIDTH || sy < 0 || sy >= SCREEN_HEIGHT) {
            st.reset(); continue;
        }
        SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, 255);
        SDL_RenderDrawLine(ren, (int)px, (int)py, (int)sx, (int)sy);
        SDL_Rect r = { (int)sx,(int)sy,2,2 };
        SDL_RenderFillRect(ren, &r);
    }
}

SDL_Texture* renderText(SDL_Renderer* ren, TTF_Font* f, const std::string& txt, SDL_Color col) {
    SDL_Surface* surf = TTF_RenderText_Blended(f, txt.c_str(), col);
    if (!surf) return nullptr;
    SDL_Texture* tx = SDL_CreateTextureFromSurface(ren, surf);
    SDL_FreeSurface(surf);
    if (tx) SDL_SetTextureBlendMode(tx, SDL_BLENDMODE_BLEND); // <-- viktigt
    return tx;
}

static void drawCircleOutline(SDL_Renderer* ren,
    int cx, int cy,
    int radius,
    int thickness,
    const SDL_Color& col)
{
    const int segments = 64;
    const float twoPI = 2.0f * PI;
    SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, col.a);

    // För varje ”radie‐lager” för att få önskad tjocklek
    for (int t = 0; t < thickness; ++t) {
        float r = radius - thickness / 2 + t + 0.5f;
        std::vector<SDL_Point> pts;
        pts.reserve(segments + 1);
        for (int i = 0; i <= segments; ++i) {
            float ang = twoPI * (float)i / segments;
            int x = cx + int(r * cosf(ang));
            int y = cy + int(r * sinf(ang));
            pts.push_back({ x,y });
        }
        SDL_RenderDrawLines(ren, pts.data(), (int)pts.size());
    }
}


void fillRoundedRect(SDL_Renderer* ren, SDL_Rect rect, int r, SDL_Color col) {
    SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, col.a);
    SDL_Rect mid = { rect.x + r, rect.y, rect.w - 2 * r, rect.h };
    SDL_RenderFillRect(ren, &mid);
    SDL_Rect side = { rect.x, rect.y + r, r, rect.h - 2 * r };
    SDL_RenderFillRect(ren, &side);
    side.x = rect.x + rect.w - r;
    SDL_RenderFillRect(ren, &side);
    for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
            if (dx * dx + dy * dy <= r * r) {
                SDL_RenderDrawPoint(ren, rect.x + r + dx, rect.y + r + dy);
                SDL_RenderDrawPoint(ren, rect.x + rect.w - r - 1 + dx, rect.y + r + dy);
                SDL_RenderDrawPoint(ren, rect.x + r + dx, rect.y + rect.h - r - 1 + dy);
                SDL_RenderDrawPoint(ren, rect.x + rect.w - r - 1 + dx, rect.y + rect.h - r - 1 + dy);
            }
        }
    }
}

// helper to draw thicker lines using multiple offsets
static void drawThickLine(SDL_Renderer* ren, int x1, int y1, int x2, int y2, int t)
{
    for (int dx = -t / 2; dx <= t / 2; ++dx) {
        for (int dy = -t / 2; dy <= t / 2; ++dy) {
            SDL_RenderDrawLine(ren, x1 + dx, y1 + dy, x2 + dx, y2 + dy);
        }
    }
}

static inline float smooth01(float x) { x = clampValue(x, 0.f, 1.f); return x * x * (3.f - 2.f * x); }
static inline float gfix(float a) { return powf(clampValue(a, 0.f, 1.f), 1.0f / 2.2f); }

// Enklare prism-vertex på enhetscirkel
static Vec3 prismVertex(int sides, int i, float r, float z) {
    float th = 2.f * PI * (float)i / (float)sides;
    return Vec3(cosf(th) * r, sinf(th) * r, z);
}


void initMenu() {
    const int numItems = 2;
    const char* labels[numItems] = { "RUN", "QUIT" };

    const int spacingX = 36;      // gap between RUN and QUIT
    const int marginBottom = 90;  // distance from bottom
    const int padX = 24, padY = 12;
    const float scale = MENU_BASE_SCALE;

    int w[numItems], h[numItems], hMax = 0;
    for (int i = 0; i < numItems; ++i) {
        TTF_SizeText(menuFont, labels[i], &w[i], &h[i]);
        w[i] = int(w[i] * scale);
        h[i] = int(h[i] * scale);
        hMax = std::max(hMax, h[i]);
    }

    // Rect widths include padding so hover hitbox fits even when zoomed
    int rw0 = w[0] + padX * 2;
    int rw1 = w[1] + padX * 2;
    int rh = hMax + padY * 2;

    int totalW = rw0 + spacingX + rw1;
    int startX = (SCREEN_WIDTH - totalW) / 2;
    int y = SCREEN_HEIGHT - marginBottom - rh;

    menuItems[0] = { { startX,             y, rw0, rh }, "RUN",  nullptr, 0.f, false };
    menuItems[1] = { { startX + rw0 + spacingX, y, rw1, rh }, "QUIT", nullptr, 0.f, false };

    // Pre-create green textures (same green as About text)
    SDL_Color green = { 80, 200,  80, 255 };
    for (int i = 0; i < numItems; ++i) {
        if (menuItems[i].texture) SDL_DestroyTexture(menuItems[i].texture);
        menuItems[i].texture = renderText(renderer, menuFont, menuItems[i].text, green);
    }
}

void renderMenu(float dt, int mX, int mY, bool mClick) {
    for (int i = 0; i < 2; ++i) {
        SDL_Point mp = { mX, mY };
        bool hov = SDL_PointInRect(&mp, &menuItems[i].rect);

        if (hov) {
            menuItems[i].hoverAnim = std::min(menuItems[i].hoverAnim + dt * 6.f, 1.f);
            if (!menuItems[i].wasHovered && hoverSound) {
                Mix_PlayChannel(-1, hoverSound, 0);
                menuItems[i].wasHovered = true;
            }
            if (mClick) {
                if (i == 0) {
                    currentState = STATE_PORTFOLIO;
                    initStaticStars();
                    currentEffectIndex = 0;
                    effectTimer = 0.f;
                    startStarTransition(0);
                }
                else if (i == 1) {
                    startExitExplosion(false);
                }
            }
        }
        else {
            menuItems[i].hoverAnim = std::max(menuItems[i].hoverAnim - dt * 6.f, 0.f);
            menuItems[i].wasHovered = false;
        }

        // Always same green as About text (no yellow fade)
        SDL_Color c = { 200, 255, 200, 255 };

        // Recreate texture in case font changed; keeps your existing pattern
        SDL_DestroyTexture(menuItems[i].texture);
        menuItems[i].texture = renderText(renderer, menuFont, menuItems[i].text, c);

        int tw = 0, th = 0;
        SDL_QueryTexture(menuItems[i].texture, NULL, NULL, &tw, &th);

        // Hover zoom + subtle bounce
        float scale = MENU_BASE_SCALE * (1.0f + 0.10f * menuItems[i].hoverAnim);
        float wave = sinf(elapsedTime * 3.f + i * 0.7f) * 5.f;

        SDL_Rect dst = {
            menuItems[i].rect.x + (menuItems[i].rect.w - int(tw * scale)) / 2,
            menuItems[i].rect.y + int(wave) + (menuItems[i].rect.h - int(th * scale)) / 2,
            int(tw * scale),
            int(th * scale)
        };

        SDL_RenderCopy(renderer, menuItems[i].texture, nullptr, &dst);
        // ta "dot-isch" + scanlines på knapptxten
        applyDotMask(renderer, dst);
        //applyScanlines(renderer, dst);


    }
}

void initPortfolio() {
    // smallStars.clear(); bigStars.clear();
    for (int i = 0; i < NUM_SMALL_STARS; ++i) smallStars.emplace_back();
    for (int i = 0; i < NUM_BIG_STARS; ++i) bigStars.emplace_back();
    initStaticStars();
    initInterference();
    currentEffectIndex = 0;
    effectTimer = 0.f;
}

void renderPlasmaPolygon(float time) {
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(0.f, 0.f, -5.f);
    glRotatef(time * 30.f, 1.f, 0.f, 0.f);
    glRotatef(time * 45.f, 0.f, 1.f, 0.f);
    glDisable(GL_TEXTURE_2D);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
    glBegin(GL_TRIANGLES);
    const int sides = 8;
    float half = 1.f;
    for (int i = 0; i < sides; ++i) {
        float th1 = i * 2.f * PI / sides;
        float th2 = (i + 1) * 2.f * PI / sides;
        float x1 = cosf(th1), y1 = sinf(th1);
        float x2 = cosf(th2), y2 = sinf(th2);
        SDL_Color c1 = plasmaColor(int((x1 + 1.f) * SCREEN_WIDTH / 2), int((y1 + 1.f) * SCREEN_HEIGHT / 2), time);
        SDL_Color c2 = plasmaColor(int((x2 + 1.f) * SCREEN_WIDTH / 2), int((y2 + 1.f) * SCREEN_HEIGHT / 2), time);
        SDL_Color c3 = plasmaColor(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, time);
        glColor3ub(c1.r, c1.g, c1.b); glVertex3f(x1, y1, half);
        glColor3ub(c2.r, c2.g, c2.b); glVertex3f(x2, y2, half);
        glColor3ub(c3.r, c3.g, c3.b); glVertex3f(0.f, 0.f, -half);
    }
    glEnd();
    glDisable(GL_CULL_FACE);
    glPopMatrix();
}

Vec3 rotateY(const Vec3& v, float a) {
    float c = cosf(a), s = sinf(a);
    return Vec3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}
Vec3 rotateX(const Vec3& v, float a) {
    float c = cosf(a), s = sinf(a);
    return Vec3(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}
SDL_Point projectPoint(const Vec3& v, int w, int h, float fov, float dist) {
    float factor = fov / (dist + v.z);
    int x = int(v.x * factor + w / 2);
    int y = int(-v.y * factor + h / 2);
    return { x,y };
}

void ensurePlasmaTexture(SDL_Renderer* ren) {
    if (!gPlasma) {
        gPlasma = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING, gPlasmaW, gPlasmaH);
        SDL_SetTextureBlendMode(gPlasma, SDL_BLENDMODE_BLEND);
        SDL_SetTextureScaleMode(gPlasma, SDL_ScaleModeLinear);
    }
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static Vec3 normalizeVec(const Vec3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len == 0.f) return Vec3(0.f, 0.f, 0.f);
    return Vec3(v.x / len, v.y / len, v.z / len);
}

const Vec3 cubeVertices[8] = {
    {-1,-1,-1},{ 1,-1,-1},{ 1, 1,-1},{-1, 1,-1},
    {-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}
};
const int cubeEdges[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

void renderWireframeCube(SDL_Renderer* ren, float ax, float ay, SDL_Color col) {
    Vec3 rot[8];
    for (int i = 0; i < 8; ++i) {
        Vec3 v = rotateY(cubeVertices[i], ay);
        rot[i] = rotateX(v, ax);
    }
    for (int i = 0; i < 12; ++i) {
        SDL_Color ec = rainbowColor(i / 12.f);
        SDL_SetRenderDrawColor(ren, ec.r, ec.g, ec.b, col.a);
        SDL_Point p1 = projectPoint(rot[cubeEdges[i][0]], SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f);
        SDL_Point p2 = projectPoint(rot[cubeEdges[i][1]], SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f);
        drawThickLine(ren, p1.x, p1.y, p2.x, p2.y, 6);
    }
}


// OpenGL version for smoother look
void renderSolidCubeGL(float ax, float ay, float time) {
    glLoadIdentity();
    glPushMatrix();
    glDisable(GL_DEPTH_TEST);
    glTranslatef(0.f, 0.f, -5.f);
    glRotatef(ax * 57.2958f, 1.f, 0.f, 0.f);
    glRotatef(ay * 57.2958f, 0.f, 1.f, 0.f);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE); // show all edges of the wireframe shape
    glLineWidth(6.f);
    drawGreyCube();
    glLineWidth(1.f);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

static void drawColoredCube(SDL_Color c) {
    glColor3ub(c.r, c.g, c.b);
    glBegin(GL_QUADS);
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);

    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glEnd();
}

// Simple grey cube used for wireframe shapes
static void drawGreyCube() {

    const float half = 1.f;
    glBegin(GL_QUADS);
    glNormal3f(0.f, 0.f, 1.f);
    glVertex3f(-half, -half, half);
    glVertex3f(half, -half, half);
    glVertex3f(half, half, half);
    glVertex3f(-half, half, half);

    glNormal3f(0.f, 0.f, -1.f);
    glVertex3f(-half, -half, -half);
    glVertex3f(-half, half, -half);
    glVertex3f(half, half, -half);
    glVertex3f(half, -half, -half);

    glNormal3f(-1.f, 0.f, 0.f);
    glVertex3f(-half, -half, -half);
    glVertex3f(-half, -half, half);
    glVertex3f(-half, half, half);
    glVertex3f(-half, half, -half);

    glNormal3f(1.f, 0.f, 0.f);
    glVertex3f(half, -half, half);
    glVertex3f(half, -half, -half);
    glVertex3f(half, half, -half);
    glVertex3f(half, half, half);

    glNormal3f(0.f, 1.f, 0.f);
    glVertex3f(-half, half, half);
    glVertex3f(half, half, half);
    glVertex3f(half, half, -half);
    glVertex3f(-half, half, -half);

    glNormal3f(0.f, -1.f, 0.f);
    glVertex3f(-half, -half, -half);
    glVertex3f(half, -half, -half);
    glVertex3f(half, -half, half);
    glVertex3f(-half, -half, half);

    glEnd();
}

void renderFilledPrismGL(int sides, float ax, float ay, float time) {
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(0.f, 0.f, -5.f);
    glRotatef(ax * 57.2958f, 1.f, 0.f, 0.f);
    glRotatef(ay * 57.2958f, 0.f, 1.f, 0.f);
    glDisable(GL_TEXTURE_2D);

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE); // show all edges of the wireframe shape
    drawSolidPrism(sides);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_LIGHTING);
    glPopMatrix();
}

void renderCylinderGL(float angle) {
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(0.f, 0.f, -6.f);
    glRotatef(angle * 57.2958f, 0.f, 1.f, 0.f);
    const int segs = 24;
    float radius = 1.f, height = 2.f;
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE); // show all edges of the wireframe shape

    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; ++i) {
        float th = 2.f * PI * i / segs;
        float x = cosf(th) * radius;
        float z = sinf(th) * radius;
        SDL_Color c = rainbowColor(i / (float)segs);
        glColor3ub(c.r, c.g, c.b);
        glVertex3f(x, height / 2, z);
        glVertex3f(x, -height / 2, z);
    }
    glEnd();

    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.f, 1.f, 0.f);
    glVertex3f(0.f, height / 2, 0.f);
    for (int i = 0; i <= segs; ++i) {
        float th = 2.f * PI * i / segs;
        glVertex3f(cosf(th) * radius, height / 2, sinf(th) * radius);
    }
    glEnd();

    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.f, -1.f, 0.f);
    glVertex3f(0.f, -height / 2, 0.f);
    for (int i = segs; i >= 0; --i) {
        float th = 2.f * PI * i / segs;
        glVertex3f(cosf(th) * radius, -height / 2, sinf(th) * radius);
    }
    glEnd();

    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();
}

void renderDotPrismGL(int sides, float ax, float ay, float time) {
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(0.f, 0.f, -5.f);
    glRotatef(ax * 57.2958f, 1.f, 0.f, 0.f);
    glRotatef(ay * 57.2958f, 0.f, 1.f, 0.f);
    glDisable(GL_TEXTURE_2D);

    glPointSize(3.f);
    glBegin(GL_POINTS);
    for (int i = 0; i < sides; ++i) {
        float th1 = i * 2.f * PI / sides;
        float th2 = (i + 1) * 2.f * PI / sides;
        for (float t = 0.f; t <= 1.f; t += 0.1f) {
            float x = cosf(th1) * (1.f - t) + cosf(th2) * t;
            float y = sinf(th1) * (1.f - t) + sinf(th2) * t;
            for (float z = -1.f; z <= 1.f; z += 0.2f) {
                SDL_Color c = rainbowColor((i + t) / sides);
                glColor3ub(c.r, c.g, c.b);
                glVertex3f(x, y, z);
            }
        }
    }
    glEnd();
    glPointSize(1.f);
    glPopMatrix();
}

// Draw a filled prism using triangles so it appears correctly on screen
static void drawSolidPrism(int sides)
{
    const float half = 1.f;
    // side faces
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < sides; ++i) {
        float th1 = i * 2.f * PI / sides;
        float th2 = (i + 1) * 2.f * PI / sides;
        float x1 = cosf(th1), y1 = sinf(th1);
        float x2 = cosf(th2), y2 = sinf(th2);
        Vec3 n = normalizeVec(cross(Vec3(x2 - x1, y2 - y1, 0.f), Vec3(-x1, -y1, half * 2.f)));
        glNormal3f(n.x, n.y, n.z);
        glVertex3f(x1, y1, half);
        glVertex3f(x1, y1, -half);
        glVertex3f(x2, y2, -half);
        glVertex3f(x2, y2, -half);
        glVertex3f(x2, y2, half);
        glVertex3f(x1, y1, half);
    }
    glEnd();
    // top face
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.f, 0.f, 1.f);
    glVertex3f(0.f, 0.f, half);
    for (int i = 0; i <= sides; ++i) {
        float th = i * 2.f * PI / sides;
        glVertex3f(cosf(th), sinf(th), half);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.f, 0.f, -1.f);
    glVertex3f(0.f, 0.f, -half);
    for (int i = sides; i >= 0; --i) {
        float th = i * 2.f * PI / sides;
        glVertex3f(cosf(th), sinf(th), -half);
    }
    glEnd();
}

// Pyramid with a square base
static void drawPyramid() {
    const float half = 1.f;
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 4; ++i) {
        float th1 = i * 0.5f * PI;
        float th2 = (i + 1) * 0.5f * PI;
        Vec3 v1{ cosf(th1), sinf(th1), -half };
        Vec3 v2{ cosf(th2), sinf(th2), -half };
        Vec3 apex{ 0.f,0.f,half };
        Vec3 n = normalizeVec(cross(Vec3(v2.x - v1.x, v2.y - v1.y, 0.f), Vec3(apex.x - v1.x, apex.y - v1.y, apex.z - v1.z)));
        glNormal3f(n.x, n.y, n.z);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
        glVertex3f(apex.x, apex.y, apex.z);
    }
    glEnd();
    glBegin(GL_QUADS);
    glNormal3f(0.f, 0.f, -1.f);
    glVertex3f(-1.f, -1.f, -half);
    glVertex3f(1.f, -1.f, -half);
    glVertex3f(1.f, 1.f, -half);
    glVertex3f(-1.f, 1.f, -half);
    glEnd();
}

// Simple cone shape
static void drawCone(int segs) {
    const float half = 1.f;
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < segs; ++i) {
        float th1 = 2.f * PI * i / segs;
        float th2 = 2.f * PI * (i + 1) / segs;
        Vec3 v1{ cosf(th1), sinf(th1), -half };
        Vec3 v2{ cosf(th2), sinf(th2), -half };
        Vec3 apex{ 0.f,0.f,half };
        Vec3 n = normalizeVec(cross(Vec3(v2.x - v1.x, v2.y - v1.y, 0.f), Vec3(apex.x - v1.x, apex.y - v1.y, apex.z - v1.z)));
        glNormal3f(n.x, n.y, n.z);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
        glVertex3f(apex.x, apex.y, apex.z);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.f, 0.f, -1.f);
    glVertex3f(0.f, 0.f, -half);
    for (int i = segs; i >= 0; --i) {
        float th = 2.f * PI * i / segs;
        glVertex3f(cosf(th), sinf(th), -half);
    }
    glEnd();
}

static const int GRID_W = 20;
static const int GRID_H = 20;
static const int CELL = 20;
static const int WORM_MAX_SCORE = 10;
static int snakeDir = 0;
static float snakeTimer = 0.f;
static SDL_Point food{ 0,0 };
static int snakeScore = 0;
static bool snakeGameOver = false;
static bool snakeWin = false;            // <-- NEW: Win condition flag
static bool snakeExploded = false;       // <-- NEW: Explosion screen flag
static float snakeExplodeTimer = 0.f;    // <-- NEW: For display timing
std::vector<SDL_Point> snake;

// --- Simple pong minigame ---
static SDL_Rect paddleLeft, paddleRight;
static float ballX = 0.f, ballY = 0.f, ballVX = 0.f, ballVY = 0.f;
static int paddleDir = 0; // -1 up, 1 down
static int pongScore = 0;
static bool pongGameOver = false;
static float pongGameOverTimer = 0.f;

// --- Tunables för fartskala ---
static inline float snakeStepInterval(int score) {
    const float BASE = 0.15f;   // startfart
    const float STEP = 0.010f;  // snabbare per poäng
    const float MINI = 0.055f;  // golv (maxfart)
    float inter = BASE - STEP * score;
    return (inter < MINI) ? MINI : inter;
}

static inline bool pointOnSnake(const std::vector<SDL_Point>& s, int x, int y) {
    for (const auto& p : s) if (p.x == x && p.y == y) return true;
    return false;
}

void initSnakeGame() {
    snake.clear();
    snake.push_back({ GRID_W / 2, GRID_H / 2 });
    snakeDir = 0;
    snakeTimer = 0.f;
    snakeScore = 0;
    snakeGameOver = false;
    snakeWin = false;
    snakeExploded = false;
    snakeExplodeTimer = 0.f;

    // placera mat inte på ormen
    do {
        food.x = rand() % GRID_W;
        food.y = rand() % GRID_H;
    } while (pointOnSnake(snake, food.x, food.y));
}

void updateSnakeGame(float dt) {
    if (snakeExploded) {
        snakeExplodeTimer += dt;
        // efter kort “panel-explosion”, gå vidare
        if (snakeExplodeTimer > 1.2f) {
            int idx = (currentEffectIndex + 1) % NUM_EFFECTS;
            startStarTransition(idx);
        }
        return;
    }
    if (snakeGameOver || snakeWin) return;

    snakeTimer += dt;
    const float interval = snakeStepInterval(snakeScore); // snabbare med poäng
    if (snakeTimer >= interval) {
        snakeTimer -= interval;

        SDL_Point head = snake.back();
        switch (snakeDir) {
        case 0: head.x++; break;
        case 1: head.y++; break;
        case 2: head.x--; break;
        case 3: head.y--; break;
        }
        // wrap
        if (head.x < 0) head.x = GRID_W - 1; else if (head.x >= GRID_W) head.x = 0;
        if (head.y < 0) head.y = GRID_H - 1; else if (head.y >= GRID_H) head.y = 0;

        // kollision med sig själv
        for (const auto& p : snake) {
            if (p.x == head.x && p.y == head.y) {
                snakeGameOver = true;
                snakeWin = false;
                snakeExploded = true;     // explosion sker i panelen (render)
                snakeExplodeTimer = 0.f;
                return;
            }
        }

        snake.push_back(head);

        // åt mat?
        if (head.x == food.x && head.y == food.y) {
            // ny mat ej på ormen
            do {
                food.x = rand() % GRID_W;
                food.y = rand() % GRID_H;
            } while (pointOnSnake(snake, food.x, food.y));

            snakeScore++;
            if (snakeScore >= WORM_MAX_SCORE) {
                snakeWin = true;
                snakeExploded = true;     // panel-explosion, sen transition
                snakeExplodeTimer = 0.f;
                return;
            }
        }
        else {
            // rör på sig
            snake.erase(snake.begin());
        }
    }
}

void renderSnakeGame(SDL_Renderer* ren) {
    // Gör spelfönstret “lite större” genom större celler lokalt
    const int CELL_LOCAL = CELL + 2; // +2 px per cell = lite större fönster

    const int playW = GRID_W * CELL_LOCAL;
    const int playH = GRID_H * CELL_LOCAL;
    const int startX = (SCREEN_WIDTH - playW) / 2;
    const int startY = (SCREEN_HEIGHT - playH) / 2;

    // Semi-transparent bakgrund i fönstret (lite mer transparent än tidigare)
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_Rect bg = { startX, startY, playW, playH };
    SDL_SetRenderDrawColor(ren, 10, 20, 35, 140);   // panel med ~55% alpha
    SDL_RenderFillRect(ren, &bg);

    // Ram
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 200);
    SDL_Rect frame = { startX - 2, startY - 2, playW + 4, playH + 4 };
    SDL_RenderDrawRect(ren, &frame);
    SDL_Rect outer = { startX - 3, startY - 3, playW + 6, playH + 6 };
    SDL_RenderDrawRect(ren, &outer);

    // Ormen (svans tonar)
    for (size_t i = 0; i < snake.size(); ++i) {
        const auto& sSeg = snake[i];
        SDL_Rect r{ startX + sSeg.x * CELL_LOCAL, startY + sSeg.y * CELL_LOCAL, CELL_LOCAL, CELL_LOCAL };
        float t = (i + 1) / (float)snake.size();
        Uint8 a = Uint8(80 + t * 175);
        SDL_SetRenderDrawColor(ren, 50, 220, 50, a);
        SDL_RenderFillRect(ren, &r);
    }

    // Mat
    SDL_Rect fr{ startX + food.x * CELL_LOCAL, startY + food.y * CELL_LOCAL, CELL_LOCAL, CELL_LOCAL };
    SDL_SetRenderDrawColor(ren, 220, 50, 50, 230);
    SDL_RenderFillRect(ren, &fr);

    // Titel + poäng
    TTF_Font* smallFont = consoleFont ? consoleFont : menuFont;

    if (SDL_Texture* t = renderText(ren, smallFont, "Worms", { 255,255,255,220 })) {
        int w, h; SDL_QueryTexture(t, nullptr, nullptr, &w, &h);
        SDL_Rect dst = { (SCREEN_WIDTH - w) / 2, startY - h - 5, w, h };
        SDL_RenderCopy(ren, t, nullptr, &dst);
        SDL_DestroyTexture(t);
    }
    {
        std::string scoreStr = "Score: " + std::to_string(snakeScore) + "/" + std::to_string(WORM_MAX_SCORE);
        if (SDL_Texture* sc = renderText(ren, smallFont, scoreStr, { 200,255,200,220 })) {
            int w, h; SDL_QueryTexture(sc, nullptr, nullptr, &w, &h);
            SDL_Rect dst = { startX, startY - h - 5, w, h };
            SDL_RenderCopy(ren, sc, nullptr, &dst);
            SDL_DestroyTexture(sc);
        }
    }

    // Panel-lokal explosion/flash (INNE i spelrutan)
    if (snakeExploded) {
        Uint8 flash = (Uint8)(255.0f * std::fabs(std::sin(snakeExplodeTimer * 18.f)));
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(ren, flash, flash, flash, 95); // svagare, bara i panelen
        SDL_RenderFillRect(ren, &bg);

        // WIN/LOSE-meddelande inne i panelen
        std::string msg = snakeWin ? "WIN" : "GAME OVER";
        SDL_Color col = snakeWin ? SDL_Color{ 0,255,0,255 } : SDL_Color{ 255,80,80,255 };
        TTF_Font* bigF = bigFont ? bigFont : menuFont;
        if (SDL_Texture* g = renderText(ren, bigF, msg, col)) {
            int w, h; SDL_QueryTexture(g, nullptr, nullptr, &w, &h);
            SDL_Rect dst = { startX + (playW - w) / 2, startY + (playH - h) / 2, w, h };
            SDL_RenderCopy(ren, g, nullptr, &dst);
            SDL_DestroyTexture(g);
        }
    }

    // återställ blendmode om du vill:
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}

//  --- Pong minigame ---
// Local FX for Pong (flash + boxed fireworks)
static float  pongFlashT = 0.f;                 // time left for grey flash on paddle hit
static bool   pongWon = false;                  // win state to show fireworks
static bool   pongFxPrimed = false;             // spawn fireworks only once
static std::vector<FireworkParticle> pongFireworks; // local fireworks just for Pong

// Layout & tuning
static const int   PONG_PADDLE_INSET = 14;      // distance from frame edge
static const Uint8 PONG_BG_ALPHA = 110;     // transparency of the black box (0..255)
static const int   WIN_SCORE = 15;      // win threshold

// Speeds (slightly slower overall; AI a bit faster than player)
static const float PONG_PLAYER_SPEED = 420.f;
static const float PONG_AI_SPEED = 460.f;   // faster AI
static const float BALL_START_VX = 380.f;   // was 420
static const float BALL_START_VY = 240.f;   // was 260
static const float BALL_MAX_SPEED = 1200.f;

// --- Fireworks helpers (boxed inside frame) ---
static void pongSpawnFireworksBursts(const SDL_Rect& frame) {
    auto randf = [](float a, float b) { return a + (b - a) * (rand() / (float)RAND_MAX); };

    int bursts = 5 + (rand() % 3);
    for (int k = 0; k < bursts; ++k) {
        float cx = randf(float(frame.x + frame.w * 0.15f), float(frame.x + frame.w * 0.85f));
        float cy = randf(float(frame.y + frame.h * 0.15f), float(frame.y + frame.h * 0.55f));
        int count = 100 + (rand() % 80);
        float base = 220.f + float(rand() % 160);

        for (int i = 0; i < count; ++i) {
            float ang = randf(0.f, 6.2831853f);
            float spd = base + float(rand() % 180);
            SDL_Color col = rainbowColor(rand() / (float)RAND_MAX);
            pongFireworks.push_back({ cx, cy, cosf(ang) * spd, sinf(ang) * spd, 2.8f, col });
        }
    }
}
static void pongUpdateFireworks(float dt, const SDL_Rect& frame) {
    const float grav = 120.f;
    for (size_t i = 0; i < pongFireworks.size(); ) {
        auto& p = pongFireworks[i];
        p.life -= dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += grav * dt;

        bool dead = (p.life <= 0.f);
        bool off = (p.x < frame.x - 40 || p.x > frame.x + frame.w + 40 ||
            p.y < frame.y - 40 || p.y > frame.y + frame.h + 40);
        if (dead || off) pongFireworks.erase(pongFireworks.begin() + (ptrdiff_t)i);
        else ++i;
    }
}
static void pongRenderFireworks(SDL_Renderer* ren, const SDL_Rect& frame) {
    SDL_RenderSetClipRect(ren, &frame);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_ADD);
    for (auto& p : pongFireworks) {
        float t = std::max(0.f, std::min(1.f, p.life / 2.8f));
        Uint8 a = (Uint8)(t * 160);
        int   r = 1 + (int)(t * 3.0f);
        SDL_SetRenderDrawColor(ren, p.color.r, p.color.g, p.color.b, a);
        drawSmallFilledCircle(ren, (int)p.x, (int)p.y, r);
    }
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
    SDL_RenderSetClipRect(ren, nullptr);
}

// Optional: small helper to (re)serve the ball. dirX = -1 (toward player) or +1 (toward AI)
static void pongServeBall(const SDL_Rect& frame, int dirX) {
    ballX = frame.x + frame.w / 2.f;
    ballY = frame.y + frame.h / 2.f;
    float vyRand = static_cast<float>((rand() % 200) - 100); // -100..+100
    ballVX = dirX * BALL_START_VX;
    ballVY = BALL_START_VY * (vyRand / 100.f);
    if (ballVY == 0.f) ballVY = 80.f; // avoid perfectly horizontal serves
}

void initPongGame() {
    const int margin = PONGGAME_MARGIN;
    const int vMargin = PONGGAME_VMARGIN;

    SDL_Rect frame = { margin, vMargin, SCREEN_WIDTH - margin * 2, SCREEN_HEIGHT - vMargin * 2 };

    // Paddles
    paddleLeft = { frame.x + PONG_PADDLE_INSET,                         frame.y + frame.h / 2 - 75, 10, 150 };
    paddleRight = { frame.x + frame.w - PONG_PADDLE_INSET - 10 /*w*/,    frame.y + frame.h / 2 - 60, 10, 120 };

    // Ball
    pongServeBall(frame, (rand() & 1) ? -1 : +1);

    // State
    paddleDir = 0;
    pongScore = 0;
    pongGameOver = false;
    pongWon = false;
    pongFxPrimed = false;
    pongFireworks.clear();
    pongFlashT = 0.f;
    pongGameOverTimer = 0.f;
}

void updatePongGame(float dt) {
    if (pongGameOver) {
        pongGameOverTimer += dt;
        // keep fireworks alive if won
        const int margin = PONGGAME_MARGIN;
        const int vMargin = PONGGAME_VMARGIN;
        SDL_Rect frame = { margin, vMargin, SCREEN_WIDTH - margin * 2, SCREEN_HEIGHT - vMargin * 2 };
        if (pongWon) pongUpdateFireworks(dt, frame);
        return;
    }

    const int margin = PONGGAME_MARGIN;
    const int vMargin = PONGGAME_VMARGIN;
    SDL_Rect frame = { margin, vMargin, SCREEN_WIDTH - margin * 2, SCREEN_HEIGHT - vMargin * 2 };

    // Player paddle
    if (paddleDir == -1)      paddleLeft.y -= (int)(PONG_PLAYER_SPEED * dt);
    else if (paddleDir == 1)  paddleLeft.y += (int)(PONG_PLAYER_SPEED * dt);
    if (paddleLeft.y < frame.y) paddleLeft.y = frame.y;
    if (paddleLeft.y + paddleLeft.h > frame.y + frame.h) paddleLeft.y = frame.y + frame.h - paddleLeft.h;

    // AI: predictive aim (a touch of anticipation)
    float anticipate = 0.18f; // look-ahead in seconds
    float targetY = ballY + ballVY * anticipate;
    if (targetY < paddleRight.y + paddleRight.h * 0.4f)        paddleRight.y -= (int)(PONG_AI_SPEED * dt);
    else if (targetY > paddleRight.y + paddleRight.h * 0.6f)   paddleRight.y += (int)(PONG_AI_SPEED * dt);
    if (paddleRight.y < frame.y) paddleRight.y = frame.y;
    if (paddleRight.y + paddleRight.h > frame.y + frame.h) paddleRight.y = frame.y + frame.h - paddleRight.h;

    // Ball rect (BEFORE collisions)
    SDL_Rect ballRect = { (int)ballX - 5, (int)ballY - 5, 10, 10 };

    // Collisions
    bool hitLeft = false, hitRight = false;

    if (SDL_HasIntersection(&ballRect, &paddleLeft) && ballVX < 0) {
        // push ball outside the paddle to avoid re-colliding next frame
        ballX = (float)(paddleLeft.x + paddleLeft.w + 6);

        ballVX = -ballVX * 1.06f; // +6% (slightly gentler)
        float rel = (ballY - (paddleLeft.y + paddleLeft.h * 0.5f)) / (paddleLeft.h * 0.5f);
        ballVY += rel * 110.f;

        // clamp speeds
        float sgnX = (ballVX < 0 ? -1.f : 1.f);
        ballVX = sgnX * std::min(std::fabs(ballVX), BALL_MAX_SPEED);
        float sgnY = (ballVY < 0 ? -1.f : 1.f);
        ballVY = sgnY * std::min(std::fabs(ballVY), BALL_MAX_SPEED);

        ++pongScore;
        if (paddleLeft.h > 40) { paddleLeft.h -= 4; paddleLeft.y += 2; }

        hitLeft = true;

        if (pongScore >= WIN_SCORE) {
            pongGameOver = true;
            pongWon = true;
            pongGameOverTimer = 0.f;
            pongFxPrimed = false; // spawn in this frame below
        }
    }

    if (SDL_HasIntersection(&ballRect, &paddleRight) && ballVX > 0) {
        ballX = (float)(paddleRight.x - 6);

        ballVX = -ballVX * 1.03f; // +3%
        float rel = (ballY - (paddleRight.y + paddleRight.h * 0.5f)) / (paddleRight.h * 0.5f);
        ballVY += rel * 95.f;

        float sgnX = (ballVX < 0 ? -1.f : 1.f);
        ballVX = sgnX * std::min(std::fabs(ballVX), BALL_MAX_SPEED);
        float sgnY = (ballVY < 0 ? -1.f : 1.f);
        ballVY = sgnY * std::min(std::fabs(ballVY), BALL_MAX_SPEED);

        hitRight = true;
    }

    // Grey flash on any paddle hit
    if (hitLeft || hitRight) pongFlashT = 0.12f;

    // Integrate ball
    ballX += ballVX * dt;
    ballY += ballVY * dt;

    // Top/bottom bounce (against frame) with tiny speed-up
    if (ballY < (float)frame.y) {
        ballY = (float)frame.y;
        ballVY = -ballVY * 1.02f;
    }
    if (ballY > (float)(frame.y + frame.h)) {
        ballY = (float)(frame.y + frame.h);
        ballVY = -ballVY * 1.02f;
    }

    // Left miss = player loses round → Game Over (keep old behavior)
    if (ballX < (float)frame.x) {
        pongGameOver = true;
        pongWon = false;
        pongGameOverTimer = 0.f;
    }

    // Right miss = AI whiffs → player gets MINUS point (penalty), then re-serve toward player
    if (ballX > (float)(frame.x + frame.w)) {
        pongScore -= 1; // penalty (can go negative)
        pongServeBall(frame, -1); // serve back toward player
    }

    // decay flash timer
    if (pongFlashT > 0.f) pongFlashT = std::max(0.f, pongFlashT - dt);

    // On win, spawn fireworks once and keep updating them
    if (pongGameOver && pongWon) {
        if (!pongFxPrimed) {
            pongFxPrimed = true;
            pongFireworks.clear();
            pongSpawnFireworksBursts(frame);
        }
        pongUpdateFireworks(dt, frame);
    }
}

// renderPongGame — shows "Restart (R)" ONLY when the game is finished

void renderPongGame(SDL_Renderer* ren, float dt) {
   
    const int margin = PONGGAME_MARGIN;
    const int vMargin = PONGGAME_VMARGIN;

    SDL_Rect frame = { margin, vMargin, SCREEN_WIDTH - margin * 2, SCREEN_HEIGHT - vMargin * 2 };

    // Background (semi-transparent black) and border
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 180);
    SDL_RenderFillRect(ren, &frame);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    SDL_RenderDrawRect(ren, &frame);

    // Grey flash overlay on paddle hit
    if (pongFlashT > 0.f) {
        float t = pongFlashT / 0.12f;                  // 0..1
        Uint8 a = (Uint8)(100 * t);                    // fade out
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(ren, 120, 120, 120, a); // soft grey
        SDL_RenderFillRect(ren, &frame);
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
    }

    // Paddles & ball
    SDL_RenderFillRect(ren, &paddleLeft);
    SDL_RenderFillRect(ren, &paddleRight);
    SDL_Rect ballRect = { (int)ballX - 5, (int)ballY - 5, 10, 10 };
    SDL_RenderFillRect(ren, &ballRect);

    // Score
    {
        std::string line = "Score: " + std::to_string(pongScore);
        SDL_Texture* t = renderText(ren, PongFont, line, { 255,255,255,255 });
        if (t) {
            int w, h; SDL_QueryTexture(t, nullptr, nullptr, &w, &h);
            SDL_Rect dst = { frame.x + 20, frame.y + 10, w, h };
            SDL_RenderCopy(ren, t, nullptr, &dst);
            SDL_DestroyTexture(t);
        }
    }

    // If the round is over, show WIN/LOSE, hints, and (on win) fireworks
    if (pongGameOver) {

        if (pongWon) {
            // one-time spawn of boxed fireworks
            if (!pongFxPrimed) {
                pongFxPrimed = true;
                pongFireworks.clear();
                pongSpawnFireworksBursts(frame);
            }
            // update & render fireworks INSIDE the frame
            pongUpdateFireworks(dt, frame);
            pongRenderFireworks(ren, frame);

            // YOU WIN text
            SDL_Texture* winTex = renderText(ren, PongFont, "YOU WIN!", { 255,255,255,255 });
            if (winTex) {
                int w, h; SDL_QueryTexture(winTex, nullptr, nullptr, &w, &h);
                SDL_Rect dst = { (SCREEN_WIDTH - w) / 2, frame.y + frame.h / 2 - h - 20, w, h };
                SDL_RenderCopy(ren, winTex, nullptr, &dst);
                SDL_DestroyTexture(winTex);
            }
        }
        else {
            // LOSE text
            SDL_Texture* loseTex = renderText(ren, PongFont, "Game Over", { 255,255,255,255 });
            if (loseTex) {
                int w, h; SDL_QueryTexture(loseTex, nullptr, nullptr, &w, &h);
                SDL_Rect dst = { (SCREEN_WIDTH - w) / 2, frame.y + frame.h / 2 - h - 20, w, h };
                SDL_RenderCopy(ren, loseTex, nullptr, &dst);
                SDL_DestroyTexture(loseTex);
            }
        }
            
        // 2) Press R to restart (steady)
        {
            SDL_Texture* rtex = renderText(ren, PongFont, "Press R to restart", { 220,220,220,255 });
            if (rtex) {
                int w, h; SDL_QueryTexture(rtex, nullptr, nullptr, &w, &h);
                SDL_Rect dst = { (SCREEN_WIDTH - w) / 2, frame.y + frame.h / 2 + 40, w, h };
                SDL_RenderCopy(ren, rtex, nullptr, &dst);
                SDL_DestroyTexture(rtex);
            }
        }
    }
}


void renderWireframePrism(SDL_Renderer* ren, int sides, float ax, float ay, SDL_Color col) {

    const float half = 1.f;
    std::vector<Vec3> top(sides), bottom(sides);
    float rad = 1.5f;
    for (int i = 0; i < sides; ++i) {
        float th = i * 2.f * PI / sides;
        float x = cosf(th) * rad;
        float y = sinf(th) * rad;
        top[i] = Vec3(x, y, half);
        bottom[i] = Vec3(x, y, -half);
    }
    for (int i = 0; i < sides; ++i) {
        top[i] = rotateX(rotateY(top[i], ay), ax);
        bottom[i] = rotateX(rotateY(bottom[i], ay), ax);
    }
    for (int i = 0; i < sides; ++i) {
        int j = (i + 1) % sides;
        SDL_Color ec = rainbowColor(i / (float)sides);
        SDL_SetRenderDrawColor(ren, ec.r, ec.g, ec.b, col.a);
        SDL_Point t1 = projectPoint(top[i], SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f);
        SDL_Point t2 = projectPoint(top[j], SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f);
        SDL_Point b1 = projectPoint(bottom[i], SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f);
        SDL_Point b2 = projectPoint(bottom[j], SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f);
        drawThickLine(ren, t1.x, t1.y, t2.x, t2.y, 6);
        drawThickLine(ren, b1.x, b1.y, b2.x, b2.y, 6);
        drawThickLine(ren, t1.x, t1.y, b1.x, b1.y, 6);
    }
}

static void renderWireframePrismDissolve(SDL_Renderer* ren,
    int sidesFrom, int sidesTo,
    float ax, float ay, float t, int thickness = 6)
{
    // liten radieskillnad under bytet -> undvik "dubbel-linje"
    float rOut = 1.50f * (1.f - 0.06f * t);
    float rIn = 1.50f * (0.94f + 0.06f * t);
    const float half = 1.f;

    auto proj = [&](const Vec3& v) { return projectPoint(v, SCREEN_WIDTH, SCREEN_HEIGHT, 500.f, 3.f); };
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    // --- Fada UT nuvarande polygon ---
    for (int i = 0; i < sidesFrom; i++) {
        int j = (i + 1) % sidesFrom;
        Vec3 t0 = rotateX(rotateY(prismVertex(sidesFrom, i, rOut, half), ay), ax);
        Vec3 t1 = rotateX(rotateY(prismVertex(sidesFrom, j, rOut, half), ay), ax);
        Vec3 b0 = rotateX(rotateY(prismVertex(sidesFrom, i, rOut, -half), ay), ax);
        Vec3 b1 = rotateX(rotateY(prismVertex(sidesFrom, j, rOut, -half), ay), ax);

        float edgePhase = (i + 0.5f) / (float)sidesFrom;     // [0..1)
        float off = (edgePhase - 0.5f) * 0.5f;               // -0.25..+0.25 (stagger)
        float w = smooth01((t - off) / 0.75f);               // hur långt denna kant har fadat
        float alpha = gfix(1.f - w);                         // gamma-korrigera
        Uint8 a = (Uint8)(alpha * 255);
        if (!a) continue;

        SDL_Color ec = rainbowColor(i / (float)sidesFrom);
        SDL_SetRenderDrawColor(ren, ec.r, ec.g, ec.b, a);
        SDL_Point pt0 = proj(t0), pt1 = proj(t1);
        SDL_Point pb0 = proj(b0), pb1 = proj(b1);
        drawThickLine(ren, pt0.x, pt0.y, pt1.x, pt1.y, thickness);
        drawThickLine(ren, pb0.x, pb0.y, pb1.x, pb1.y, thickness);
        drawThickLine(ren, pt0.x, pt0.y, pb0.x, pb0.y, thickness);
    }

    // --- Fada IN nästa polygon ---
    for (int i = 0; i < sidesTo; i++) {
        int j = (i + 1) % sidesTo;
        Vec3 t0 = rotateX(rotateY(prismVertex(sidesTo, i, rIn, half), ay), ax);
        Vec3 t1 = rotateX(rotateY(prismVertex(sidesTo, j, rIn, half), ay), ax);
        Vec3 b0 = rotateX(rotateY(prismVertex(sidesTo, i, rIn, -half), ay), ax);
        Vec3 b1 = rotateX(rotateY(prismVertex(sidesTo, j, rIn, -half), ay), ax);

        float edgePhase = (i + 0.5f) / (float)sidesTo;
        float off = (edgePhase - 0.5f) * 0.5f;
        float w = smooth01((t - (0.15f + off)) / 0.75f);     // lite senare start
        float alpha = gfix(w);
        Uint8 a = (Uint8)(alpha * 255);
        if (!a) continue;

        SDL_Color ec = rainbowColor(i / (float)sidesTo);
        SDL_SetRenderDrawColor(ren, ec.r, ec.g, ec.b, a);
        SDL_Point pt0 = proj(t0), pt1 = proj(t1);
        SDL_Point pb0 = proj(b0), pb1 = proj(b1);
        drawThickLine(ren, pt0.x, pt0.y, pt1.x, pt1.y, thickness);
        drawThickLine(ren, pb0.x, pb0.y, pb1.x, pb1.y, thickness);
        drawThickLine(ren, pt0.x, pt0.y, pb0.x, pb0.y, thickness);
    }

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}

SDL_Color rainbowColor(float t) {
    float r = sinf(t * 2.f * PI) * 0.5f + 0.5f;
    float g = sinf(t * 2.f * PI + 2.094f) * 0.5f + 0.5f;
    float b = sinf(t * 2.f * PI + 4.188f) * 0.5f + 0.5f;
    return SDL_Color{ Uint8(r * 255),Uint8(g * 255),Uint8(b * 255),255 };
}


SDL_Color plasmaColor(int x, int y, float time) {
    float cx = x / (float)SCREEN_WIDTH;
    float cy = y / (float)SCREEN_HEIGHT;
    float v1 = sinf(cx * 10.f + time);
    float v2 = cosf(cy * 10.f + time * 1.5f);
    float v3 = sinf((cx + cy) * 13.f + time * 0.8f);
    float v4 = sinf(hypotf(cx, cy) * 15.f + time * 1.2f);
    float v5 = cosf((cx - cy) * 9.f + time * 1.3f);
    float v = (v1 + v2 + v3 + v4 + v5) / 5.f;
    v = (v + 1.f) * 0.5f;
    Uint8 r = Uint8((sinf(v * PI) * 127.f) + 128.f);
    Uint8 g = Uint8((sinf(v * PI + 2.094f) * 127.f) + 128.f);
    Uint8 b = Uint8((sinf(v * PI + 4.188f) * 127.f) + 128.f);
    return SDL_Color{ r, g, b, 255 };
}

void renderPlasma(SDL_Renderer* ren, float time) {
    const int blockSize = 60;
    float ox = sinf(time * 2.0f) * 300.f + cosf(time * 0.7f) * 120.f;
    float oy = cosf(time * 1.7f) * 150.f + sinf(time * 0.5f) * 90.f;
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
    for (int y = 0; y < SCREEN_HEIGHT; y += blockSize) {
        for (int x = 0; x < SCREEN_WIDTH; x += blockSize) {
            SDL_Color c = plasmaColor(int(x + ox), int(y + oy), time * 6.f);
            SDL_SetRenderDrawColor(ren, c.r, c.g, c.b, 10);
            SDL_Rect r = { x,y,blockSize,blockSize };
            SDL_RenderFillRect(ren, &r);
        }
    }
}

void renderFractalZoom(SDL_Renderer* ren, float dt)
{
    if (!mandelbrotShader || !mandelbrotVAO) return;

    glUseProgram(mandelbrotShader);

    struct Target { double x, y; };
    static const Target targets[] = {
        { -0.743643887037151, 0.131825904205330 },
        { -0.743643700000000, 0.131825910000000 },
        { -0.743643530000000, 0.131825880000000 }
    };
    constexpr int numTargets = int(sizeof(targets) / sizeof(targets[0]));

    // Slow down the zooming and linger a bit longer on each target so
    // the fractal "arms" have time to unfold nicely.
    const float  flyTime = 3.0f;          // was 1.2f – slower travel between targets
    const float  holdTime = 1.0f;         // was 0.3f – short pause at each stop
    const double zoomFactorPerFly = 0.5;  // gentler zoom progression
    const double minZoom = 8e-6;

    enum Phase { Fly, Hold };
    static bool   firstRun = true;
    static Phase  phase = Fly;
    static int    currentTarget = 0;
    static float  phaseTime = 0.f;

    static double centerX, centerY;
    static double startX, startY;
    static double targetX, targetY;
    static double zoom, startZoom, endZoom;

    if (firstRun) {
        firstRun = false;
        currentTarget = 0;
        phase = Fly;
        phaseTime = 0.f;

        centerX = startX = targets[0].x;
        centerY = startY = targets[0].y;
        targetX = targets[1].x;
        targetY = targets[1].y;

        startZoom = 1.5e-3;
        endZoom = startZoom * zoomFactorPerFly;
        zoom = startZoom;
    }

    // Previously the flight phase multiplied dt by 2.5 which made the
    // animation race.  Using a smaller multiplier gives a calmer motion.
    phaseTime += (phase == Fly ? dt * 0.6f : dt);

    auto ease = [](float t) {
        if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;
        return t * t * (3.f - 2.f * t);
        };

    // --- Viewport vi ritar Mandelbrot i ---
    const int viewW = 800, viewH = 600;
    const int viewX = (SCREEN_WIDTH - viewW) / 2;
    const int viewY = (SCREEN_HEIGHT - viewH) / 2;
    const int glY = SCREEN_HEIGHT - viewY - viewH;

    // [FIX] aspect som används även i shadern (uv.x *= aspect)
    const double aspect = double(viewW) / double(viewH);

    if (phase == Fly)
    {
        float t = ease(phaseTime / flyTime);

        // Lerp center and geometric zoom
        centerX = startX + (targetX - startX) * (double)t;
        centerY = startY + (targetY - startY) * (double)t;
        zoom = startZoom * pow(endZoom / startZoom, (double)t);
        if (zoom < minZoom) zoom = minZoom;

        if (t >= 1.f) { phase = Hold; phaseTime = 0.f; }
    }
    else // Hold
    {
        if (phaseTime >= holdTime) {
            ++currentTarget;
            if (currentTarget >= numTargets) {
                firstRun = true;
                glUseProgram(0);
                startStarTransition((currentEffectIndex + 1) % NUM_EFFECTS);
                return;
            }
            startX = centerX;  startY = centerY;
            targetX = targets[currentTarget].x;
            targetY = targets[currentTarget].y;

            startZoom = zoom;
            endZoom = std::max(minZoom, startZoom * zoomFactorPerFly);

            phase = Fly;
            phaseTime = 0.f;
        }
    }

    // [FIX] Orbit helt AV (för att utesluta att det är boven)
    double animX = centerX;
    double animY = centerY;

    SDL_RenderFlush(ren);
    glDisable(GL_DEPTH_TEST);

    glEnable(GL_SCISSOR_TEST);
    glScissor(viewX, glY, viewW, viewH);
    glViewport(viewX, glY, viewW, viewH);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    // Uniforms
    glUniform1f(glGetUniformLocation(mandelbrotShader, "uZoom"), (float)zoom);
    glUniform2f(glGetUniformLocation(mandelbrotShader, "uCenter"), (float)animX, (float)animY);
    glUniform2i(glGetUniformLocation(mandelbrotShader, "uResolution"), viewW, viewH);

    int maxIter = 600;
    glUniform1i(glGetUniformLocation(mandelbrotShader, "uMaxIter"), maxIter);

    glBindVertexArray(mandelbrotVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    glDisable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void updateInterference(float dt)
{
    // jämna faser som loopar snyggt (wrap är kontinuerlig pga cos/sin)
    // Use a full 2π wrap so the trigonometric functions loop smoothly
    // without the visible jump that occurred when the phase wrapped at π.
    interPhaseA = fmodf(interPhaseA + dt * 0.35f, 2.f * PI);
    interPhaseB = fmodf(interPhaseB + dt * 0.23f, 2.f * PI);

    const float cx = SCREEN_WIDTH * 0.5f;
    const float cy = SCREEN_HEIGHT * 0.5f;

    // uppdatera mjuk bana + lätt pulserande radie
    for (auto& c : interferenceCircles) {
        const float ax = (c.white ? interPhaseA : interPhaseB);
        const float ay = (c.white ? interPhaseB : interPhaseA);

        c.x = cx + cosf(ax) * (SCREEN_WIDTH * 0.35f);
        c.y = cy + sinf(ay) * (SCREEN_HEIGHT * 0.27f);

        c.r = 140.f + 40.f * (c.white ? sinf(ax * 0.7f) : cosf(ay * 0.6f));
    }
}

void renderInterference(SDL_Renderer* ren)
{
    // vi växlar blend-mode mellan grupperna
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    constexpr int ringCount = 28;
    constexpr int ringThickness = 20;
    const int baseR = std::max(SCREEN_WIDTH, SCREEN_HEIGHT);

    const int groupsToDraw = std::min<int>(2, (int)interferenceCircles.size());
    for (int i = 0; i < groupsToDraw; ++i) {
        auto& c = interferenceCircles[i];

        const bool isWhite = c.white;
        const SDL_Color col = isWhite
            ? SDL_Color{ 255,255,255,150 }
        : SDL_Color{ 0, 0, 0,255 };

        // vit grupp får ADD för mer glow/kontrast
        SDL_SetRenderDrawBlendMode(ren, isWhite ? SDL_BLENDMODE_ADD : SDL_BLENDMODE_BLEND);

        const float step = float(baseR) / float(ringCount);
        for (int j = 0; j < ringCount; ++j) {
            const int r = int(c.r + j * step);
            drawCircleOutline(ren, int(c.x), int(c.y), r, ringThickness, col);
        }
    }

    // återställ
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}

void updateFireworks(float dt)
{
    nextBurst -= dt;
    if (!exitExplosion && nextBurst <= 0.f) {
        nextBurst = 1.f;
        const float cx = static_cast<float>(rand() % SCREEN_WIDTH);
        const float cy = static_cast<float>(rand() % (SCREEN_HEIGHT / 4));

        for (int i = 0; i < 120; ++i) {
            const float ang = (rand() / (float)RAND_MAX) * 2.f * PI;
            const float spd = 50.f + static_cast<float>(rand() % 150);
            const SDL_Color col = rainbowColor(rand() / (float)RAND_MAX);

            fireworks.push_back(FireworkParticle{
                cx, cy,
                cosf(ang) * spd, sinf(ang) * spd,
                3.5f, col
                });
        }
    }

    for (size_t i = 0; i < fireworks.size(); ) {
        auto& p = fireworks[i];
        p.life -= dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += 120.f * dt;

        if (p.life <= 0.f) {
            fireworks.erase(fireworks.begin() + (ptrdiff_t)i);
        }
        else {
            ++i;
        }
    }
}

static void drawSmallFilledCircle(SDL_Renderer* ren, int cx, int cy, int radius) {
    // Dra horisontella sträckor för varje y‐rad
    for (int dy = -radius; dy <= radius; ++dy) {
        int dx = int(sqrtf(float(radius * radius - dy * dy)));
        SDL_RenderDrawLine(ren, cx - dx, cy + dy, cx + dx, cy + dy);
    }
}

void renderFireworks(SDL_Renderer* ren, float dt) {
    updateFireworks(dt);

    // Använd ADD-blend för glow
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_ADD);

    for (auto& p : fireworks) {
        // Livstidsbaserad alfa och radie
        float lifeFrac = p.life / 3.5f;  // 1.0 → nystart, 0.0 → död
        Uint8 a = Uint8(lifeFrac * 155);
        int r = 1 + int(lifeFrac * 3.0f);

        SDL_SetRenderDrawColor(ren, p.color.r, p.color.g, p.color.b, a);
        drawSmallFilledCircle(ren, int(p.x), int(p.y), r);
    }

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}

void renderLogoWithReflection(SDL_Renderer* ren, SDL_Texture* logo, int baseX) {
    if (!logo) return;
    int w, h; SDL_QueryTexture(logo, nullptr, nullptr, &w, &h);
    Uint32 t = SDL_GetTicks();
    float time = t / 500.f;
    int baseY = 100 + int(10.f * sinf(time * 2.f));
    SDL_Rect logoRect = { baseX,baseY,w,h };
    SDL_RenderCopy(ren, logo, nullptr, &logoRect);
    int refH = h / 2;
    SDL_Texture* tgt = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, w, h);
    SDL_SetTextureBlendMode(tgt, SDL_BLENDMODE_BLEND);
    SDL_SetRenderTarget(ren, tgt);
    SDL_RenderClear(ren);
    SDL_RenderCopy(ren, logo, nullptr, nullptr);
    SDL_SetRenderTarget(ren, nullptr);
    for (int i = 0; i < refH; ++i) {
        float wave = sinf(i * 0.15f + time * 3.f) * 3.f + cosf(i * 0.2f + time * 2.5f) * 2.f;
        SDL_Rect src = { 0,h - i - 1,w,1 };
        SDL_Rect dst = { baseX + int(wave), baseY + h + i, w,1 };
        Uint8 alpha = Uint8(100 - (100 * i / refH));
        SDL_SetTextureAlphaMod(tgt, alpha);
        SDL_RenderCopy(ren, tgt, &src, &dst);
    }
    SDL_DestroyTexture(tgt);
}

void renderAboutC64Typewriter(SDL_Renderer* ren, float dt) {
    SDL_RenderSetClipRect(ren, nullptr);

    static bool init = false;
    static std::string flat;           // slutlig text (korrekt)
    static std::string script;         // med felskrivningar, backspace och upprepningar
    static std::vector<float> tEvent;  // kumulativ tid per script-händelse
    static float startTime = 0.f;

    if (!init) {
        init = true;

        // --- Bygg texten (i versaler) ---
        std::vector<std::string> lines = {
            "*** CAP. LOG SCHY-DEV 2025.08.13 ***",
            "",
            "I AM MATS.9000 - A GRUMPY OLD, BUT SOMEWHAT A KIND OF NEW DEVELOPER",
            "PASSIONATE ABOUT BOTH CLASSIC AND MODERN PROGRAMMING.",
            "I LIKE CREATE ALL KIND OF SOLUTIONS. PREFERELY IN C#, C++, C and JAVA.",
            "",
            "IN THE GOOD OLD DAYS OF the 90ths, COMPUTING LED ME TO EXPLORE",
            "AMIGA ASSEMBLY, COBOL AND SOME OTHER WIERD OLD LANGUAGES. ",
            "THIS DEMO SHOWCASES SDL2 GRFX, OpenGL WITH SOME OLD SCHOOL CHARM",
            "PLEASE ENJOY A 3D-WIREFRAME WITH PLASMA WALLS, MANDELBROT FRACTAL",
            "AND SOME INTERFERENCES AND MORE. IT'S SO MUCH FUN, DONT YOU THINK!",
            "SIT BACK & ENJOY!",
            "ATDT 463613XX46 - USING:Z:MODEM",
            "",
            "PRESS RUN TO START. PRESS QUIT TO EXIT."
        };
        for (auto& s : lines)
            std::transform(s.begin(), s.end(), s.begin(), ::toupper);

        std::ostringstream oss;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i) oss << '\n';
            oss << lines[i];
        }
        flat = oss.str();

        startTime = elapsedTime;

        // --- Generera 'script' med små fel/upprepningar + tid per händelse ---
        script.clear();
        tEvent.clear();
        script.reserve(flat.size() * 2);
        tEvent.reserve(flat.size() * 2);

        const float CPS = 14.f;                 // basfart (tecken/sek)
        const float base_dt = 1.f / CPS;
        float tsum = 0.f;

        auto isWordChar = [](char c) {
            return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
            };
        auto randomWrong = []() -> char {
            const char misc[] = { '.', ',', ';', ':', '-', '*', '#', '+' };
            if (rand() % 2) return char('@' + (rand() % 26));
            if (rand() % 2) return char('0' + (rand() % 10));
            return misc[rand() % (int)(sizeof(misc))];
            };

        for (char ch : flat) {
            const bool word = isWordChar(ch);

            // Liten chans till upprepning (stamning)
            if (word && (rand() % 100) < 3) {
                script.push_back(ch);
                tsum += base_dt;                 // normal tick
                tEvent.push_back(tsum);
            }

            // Liten chans till felskrivning: fel tecken + kort paus + backspace
            if (word && (rand() % 100) < 4) {
                char wrong = randomWrong();
                script.push_back(wrong);
                tsum += base_dt * 0.8f;         // snabbt fel
                tEvent.push_back(tsum);

                tsum += base_dt * 0.4f;         // "oj"-paus
                script.push_back('\b');         // backspace
                tEvent.push_back(tsum);
            }

            // Avsett tecken + ev. paus beroende på tecken
            script.push_back(ch);
            float dtc = base_dt;
            if (ch == ' ')                      dtc += 0.02f;
            if (ch == ',' || ch == ';' || ch == ':') dtc += 0.10f;
            if (ch == '.' || ch == '!' || ch == '?') dtc += 0.22f;
            if (ch == '\n')                     dtc += 0.28f;

            tsum += dtc;
            tEvent.push_back(tsum);
        }
    }

    // --- Hur många script-händelser ska visas just nu? ---
    float tNow = elapsedTime - startTime;
    size_t shown = (size_t)(std::upper_bound(tEvent.begin(), tEvent.end(), tNow) - tEvent.begin());
    if (shown > script.size()) shown = script.size();

    // --- Bygg synliga rader genom att tolka \n och \b ---
    std::vector<std::string> outLines;
    outLines.emplace_back(); // minst en rad
    for (size_t i = 0; i < shown; ++i) {
        char c = script[i];
        if (c == '\n') {
            outLines.emplace_back();
        }
        else if (c == '\b') {
            if (!outLines.back().empty()) outLines.back().pop_back();
        }
        else {
            outLines.back().push_back(c);
        }
    }

    // --- Rendera ---
    TTF_Font* f = consoleFont ? consoleFont : menuFont;
    const int lineH = TTF_FontHeight(f);

    const int maxWidth = int(SCREEN_WIDTH * 0.70f);
    const int startX = (SCREEN_WIDTH - maxWidth) / 2 + 200;
    int y = SCREEN_HEIGHT / 2 - 340;

    int cursorX = startX, cursorY = y;

    for (size_t i = 0; i < outLines.size(); ++i) {
        const std::string& line = outLines[i];
        SDL_Texture* t = renderTextCRT(ren, f, line, SDL_Color{ 200,255,200,255 });
        int w = 0, h = 0;
        if (t) {
            SDL_QueryTexture(t, nullptr, nullptr, &w, &h);
            SDL_Rect dst{ startX, y, w, h };
            SDL_RenderCopy(ren, t, nullptr, &dst);
            applyScanlines(ren, dst, 2);
            SDL_DestroyTexture(t);

        }
        if (i + 1 == outLines.size()) { // sista raden → cursorpos
            cursorX = startX + w;
            cursorY = y;
        }
        y += lineH + 6;
    }

    // --- Blinkande C64-block-cursor ---
    bool showCursor = ((int)(elapsedTime * 2.0f) % 2) == 0; // ~2 Hz
    if (showCursor) {
        int chW, chH;
        TTF_SizeText(f, "M", &chW, &chH);
        SDL_Rect cur{ cursorX + 4, cursorY + chH - (chH - 4), chW / 2, chH - 6 };
        SDL_SetRenderDrawColor(ren, 120, 255, 120, 255);
        SDL_RenderFillRect(ren, &cur);
    }

    int areaY = SCREEN_HEIGHT / 2 - 340;
    int areaH = int(outLines.size()) * (lineH + 6);
    SDL_Rect aboutArea{ startX, areaY, maxWidth, areaH };

}


void startExitExplosion(bool returnToMenu, bool nextEffect, int nextIndex)
{
    fireworks.clear();

    auto spawn = [](float x, float y, SDL_Color col, int count, float speedBase) {
        for (int i = 0; i < count; ++i) {
            float ang = (rand() / (float)RAND_MAX) * 2.f * PI;
            float spd = speedBase + static_cast<float>(rand() % 300);

            fireworks.push_back({ x, y, cosf(ang) * spd, sinf(ang) * spd, 5.5f, col });

        }
        };

    if (!returnToMenu && !nextEffect) {
        // big central burst when exiting from menu
        for (int k = 0; k < 10; ++k) {
            SDL_Color col = rainbowColor(rand() / (float)RAND_MAX);
            spawn(SCREEN_WIDTH / 2.f, SCREEN_HEIGHT / 2.f, col, 200, 400.f);
        }
    }
    else {
        for (int j = 0; j < 6; ++j) {
            float cx = SCREEN_WIDTH / 2.f + static_cast<float>((rand() % 400) - 200);
            float cy = SCREEN_HEIGHT / 3.f + static_cast<float>((rand() % 200) - 100);
            SDL_Color col = rainbowColor(rand() / (float)RAND_MAX);
            spawn(cx, cy, col, 120, 250.f);
        }
    }
    nextBurst = 0.6f;

    if (bangSound) Mix_PlayChannel(-1, bangSound, 0);
    exitExplosion = true;
    exitTimer = 0.f;
    explosionReturnToMenu = returnToMenu;
    explosionAdvanceEffect = nextEffect;
    explosionNextIndex = nextIndex;
}

bool renderPortfolioEffect(SDL_Renderer* ren, float deltaTime) {
    bool usedGL = false;

    switch (currentPortfolioSubState) {

    case VIEW_WIREFRAME_CUBE: {
        renderStars(ren, smallStars, { 180,180,255,255 });
        renderStars(ren, bigStars, { 255,255,255,255 });
        renderStaticStars(ren);

        // --- fly-out progress & parametrar ---
        float flyDist = 2.8f;               // normal dist i projectPointPanel
        Uint8 aPlasma = static_cast<Uint8>(230);
        Uint8 aLines = WF_LINE_ALPHA;

        if (WF_FlyOut) {
            WF_FlyT += deltaTime;
            float t = clampValue(WF_FlyT / WF_FlyDur, 0.f, 1.f);
            float e = easeOutCubic(t);
            flyDist = 2.f + 62.f * e;                // skjut inåt skärmen
            aPlasma = (Uint8)(230 * (1.f - e));      // fadda plasma
            aLines = (Uint8)(WF_LINE_ALPHA * (1.f - e));
            if (t >= 1.f) {
                WF_FlyOut = false; WF_FlyT = 0.f;
                currentEffectIndex = (currentEffectIndex + 1) % NUM_EFFECTS;
                startPortfolioEffect(effectSequence[currentEffectIndex]);
                return usedGL;
            }
        }

        // Panel/clip
        const int frameW = SCREEN_WIDTH - 2 * PONG_MARGIN;
        const int frameH = SCREEN_HEIGHT - 2 * PONG_VMARGIN;
        const int frameX = (SCREEN_WIDTH - frameW) / 2;
        const int frameY = (SCREEN_HEIGHT - frameH) / 2;
        SDL_Rect frameRect{ frameX, frameY, frameW, frameH };

        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(ren, 0, 0, 0, 0);
        SDL_RenderFillRect(ren, &frameRect);
        SDL_RenderSetClipRect(ren, &frameRect);

        // --- MORPH-ring (värld -> roterat)
        const int shapes[] = { 4, 3, 5, 6 };
        const int N = (int)(sizeof(shapes) / sizeof(shapes[0]));
        float tCycle = fmodf(effectTimer, SHAPE_DURATION) / SHAPE_DURATION;
        int   cur = (int)(fmodf(effectTimer, SHAPE_DURATION * N) / SHAPE_DURATION) % N;
        int   nxt = (cur + 1) % N;

        auto smooth = [](float x) { x = clampValue(x, 0.f, 1.f); return x * x * (3.f - 2.f * x); };
        const float morphStart = 0.60f, morphEnd = 1.00f;
        float morphT = smooth((tCycle - morphStart) / (morphEnd - morphStart));

        const int   k0 = shapes[cur], k1 = shapes[nxt];
        const float rad = 1.5f, half = 1.0f;
        const int   samples = std::max(k0, k1) * 6;

        std::vector<Vec3> top, bot;
        buildMorphRing(k0, k1, morphT, samples, rad, half, cubeAngleX, cubeAngleY, top, bot);

        // --- PLASMA: se till att texturen finns och är uppdaterad (A=255)
        ensurePlasmaTexture(ren);
        SDL_SetTextureBlendMode(gPlasma, SDL_BLENDMODE_NONE);          // texturen själv ska inte blenda
        updatePlasmaTexture(gPlasma, elapsedTime, 3.8f, 0.22f, 0.12f); // fyll RGBA med A=255

        // --- Bygg och sortera sidytor (Painter's back->front för korrekt alfa)
        struct Face {
            SDL_Point p0, p1, p2, p3;
            float     zavg;       // större = längre bort (dist + z i projektionen)
            float     u0, u1;
        };
        std::vector<Face> faces;
        faces.reserve(samples);

        for (int i = 0; i < samples; ++i) {
            int j = (i + 1) % samples;

            Face f{};
            f.p0 = projectPointPanel(top[i], frameRect, 500.f, flyDist);
            f.p1 = projectPointPanel(top[j], frameRect, 500.f, flyDist);
            f.p2 = projectPointPanel(bot[j], frameRect, 500.f, flyDist);
            f.p3 = projectPointPanel(bot[i], frameRect, 500.f, flyDist);

            f.zavg = (top[i].z + top[j].z + bot[j].z + bot[i].z) * 0.25f; // större = längre bort
            f.u0 = i / (float)samples;
            f.u1 = j / (float)samples;
            faces.push_back(f);
        }
        std::sort(faces.begin(), faces.end(),
            [](const Face& a, const Face& b) { return a.zavg > b.zavg; }); // back->front

        // --- Rita fyllda väggar (två tri/quad), med jämn alfa i vertexfärg
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

        for (const Face& f : faces) {
            SDL_Vertex v[6];
            auto setV = [](SDL_Vertex& vv, float x, float y, float u, float tv, Uint8 a) {
                vv.position.x = x; vv.position.y = y;
                vv.color = SDL_Color{ 255,255,255,a }; // multiplicera inte ned färgen
                vv.tex_coord.x = u; vv.tex_coord.y = tv;
                };

            setV(v[0], (float)f.p0.x, (float)f.p0.y, f.u0, 0.f, aPlasma);
            setV(v[1], (float)f.p1.x, (float)f.p1.y, f.u1, 0.f, aPlasma);
            setV(v[2], (float)f.p2.x, (float)f.p2.y, f.u1, 1.f, aPlasma);

            setV(v[3], (float)f.p0.x, (float)f.p0.y, f.u0, 0.f, aPlasma);
            setV(v[4], (float)f.p2.x, (float)f.p2.y, f.u1, 1.f, aPlasma);
            setV(v[5], (float)f.p3.x, (float)f.p3.y, f.u0, 1.f, aPlasma);

            SDL_RenderGeometry(ren, gPlasma, v, 6, nullptr, 0);
        }

        // --- Wireframe-kanter ovanpå
        if (WF_LINE_THICKNESS > 0 && aLines > 0) {
            for (int k = 0; k < samples; ++k) {
                int kNext = (k + 1) % samples;
                SDL_Point t1 = projectPointPanel(top[k], frameRect, 500.f, flyDist);
                SDL_Point t2 = projectPointPanel(top[kNext], frameRect, 500.f, flyDist);
                SDL_Point b1 = projectPointPanel(bot[k], frameRect, 500.f, flyDist);
                SDL_Point b2 = projectPointPanel(bot[kNext], frameRect, 500.f, flyDist);

                SDL_Color ec = rainbowColor(k / (float)samples);
                SDL_SetRenderDrawColor(ren, ec.r, ec.g, ec.b, aLines);

                if (WF_LINE_THICKNESS == 1) {
                    SDL_RenderDrawLine(ren, t1.x, t1.y, t2.x, t2.y);
                    SDL_RenderDrawLine(ren, b1.x, b1.y, b2.x, b2.y);
                    SDL_RenderDrawLine(ren, t1.x, t1.y, b1.x, b1.y);
                }
                else {
                    drawThickLine(ren, t1.x, t1.y, t2.x, t2.y, WF_LINE_THICKNESS);
                    drawThickLine(ren, b1.x, b1.y, b2.x, b2.y, WF_LINE_THICKNESS);
                    drawThickLine(ren, t1.x, t1.y, b1.x, b1.y, WF_LINE_THICKNESS);
                }
            }
        }

        SDL_RenderSetClipRect(ren, nullptr);
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
        break;
    }

    
    case VIEW_INTERFERENCE:
        updateInterference(deltaTime);
        renderInterference(ren);
        break;

    case VIEW_FRACTAL_ZOOM:
        renderFractalZoom(ren, deltaTime);
        usedGL = true;
        break;

    case VIEW_SNAKE_GAME:
        updateSnakeGame(deltaTime);
        renderSnakeGame(ren);
        break;

    case VIEW_PONG_GAME:
        updatePongGame(deltaTime);
        renderPongGame(ren, deltaTime);
        break;


    case VIEW_C64_10PRINT: {
        static bool c64Started = false;
        if (!c64Started) {
            ensureGLContextCurrent();
            startC64Window();
            c64Started = true;
        }

        ensureGLContextCurrent();
        updateC64Window(deltaTime);
        renderC64Window(SCREEN_WIDTH, SCREEN_HEIGHT);
        usedGL = true;

        if (c64WindowIsDone()) {
            c64Started = false;
            int idx = (currentEffectIndex + 1) % NUM_EFFECTS;
            startStarTransition(idx);
        }
        break;
    }

    default: break;
    }

    return usedGL;
}


static std::vector<float> lutRad, lutSum, lutSinX, lutCosY;
static bool plasmaLUTReady = false;

static void initPlasmaLUTs() {
    if (plasmaLUTReady) return;
    lutRad.resize(gPlasmaW * gPlasmaH);
    lutSum.resize(gPlasmaW * gPlasmaH);
    lutSinX.resize(gPlasmaW);
    lutCosY.resize(gPlasmaH);

    for (int y = 0; y < gPlasmaH; ++y) {
        float ny = (float)y / (float)gPlasmaH;
        for (int x = 0; x < gPlasmaW; ++x) {
            float nx = (float)x / (float)gPlasmaW;
            float dx = nx - 0.5f, dy = ny - 0.5f;
            lutRad[y * gPlasmaW + x] = sqrtf(dx * dx + dy * dy);   // hypot
            lutSum[y * gPlasmaW + x] = nx + ny;                    // (cx+cy)
        }
    }
    plasmaLUTReady = true;
}

static void updatePlasmaTexture(SDL_Texture* tex, float time,
    float speed, float scrollU, float scrollV)
{
    if (!tex) return;
    void* pixels; int pitch;
    if (SDL_LockTexture(tex, nullptr, &pixels, &pitch) != 0) return;

    Uint8* base = static_cast<Uint8*>(pixels);
    const float tt = time * speed;

    for (int y = 0; y < gPlasmaH; ++y) {
        Uint32* row = reinterpret_cast<Uint32*>(base + y * pitch);
        const float cy = (y / (float)gPlasmaH) + scrollV * time;

        for (int x = 0; x < gPlasmaW; ++x) {
            const float cx = (x / (float)gPlasmaW) + scrollU * time;

            float v1 = sinf((cx * 12.0f) + tt * 0.90f);
            float v2 = cosf((cy * 11.0f) - tt * 1.10f);
            float v3 = sinf(((cx + cy) * 9.0f) + tt * 0.55f);
            float r = (fabsf(cx - 0.5f) + fabsf(cy - 0.5f));
            float v4 = sinf((r * 14.0f) + tt * 0.70f);

            float v = (v1 + v2 + v3 + v4) * 0.25f;
            v = (v + 1.f) * 0.5f;

            Uint8 r8 = (Uint8)((sinf(v * PI) * 127.f) + 128.f);
            Uint8 g8 = (Uint8)((sinf(v * PI + 2.094f) * 127.f) + 128.f);
            Uint8 b8 = (Uint8)((sinf(v * PI + 4.188f) * 127.f) + 128.f);

            row[x] = SDL_MapRGBA(gARGB, r8, g8, b8, 255);
        }
    }
    SDL_UnlockTexture(tex);
}



static void renderPrismSidesWithTexture(SDL_Renderer* ren, SDL_Texture* tex,
    int sides, float ax, float ay, const SDL_Rect& panel, Uint8 alpha)
{
    const float half = 1.f, rad = 1.5f;
    const int OFFSET_X = 200;                // flytta höger
  
    // Förprojektera och samla sidor
    struct Face {
        SDL_Point p0, p1, p2, p3;  // screen
        float zavg;                // för sortering
        float u0, u1;              // för enkel UV
    };
    std::vector<Face> faces;
    faces.reserve(sides);

    std::vector<Vec3> top(sides), bot(sides);
    for (int i = 0; i < sides; ++i) {
        float th = 2.f * PI * i / sides;
        Vec3 t{ cosf(th) * rad, sinf(th) * rad,  half };
        Vec3 b{ cosf(th) * rad, sinf(th) * rad, -half };
        top[i] = rotateX(rotateY(t, ay), ax);
        bot[i] = rotateX(rotateY(b, ay), ax);
    }

    for (int i = 0; i < sides; ++i) {
        int j = (i + 1) % sides;
        Face f{};
        f.p0 = projectPointPanel(top[i], panel);
        f.p1 = projectPointPanel(top[j], panel);
        f.p2 = projectPointPanel(bot[j], panel);
        f.p3 = projectPointPanel(bot[i], panel);
        f.zavg = (top[i].z + top[j].z + bot[j].z + bot[i].z) * 0.25f; // större = längre bort
        f.u0 = (float)i / sides;
        f.u1 = (float)j / sides;
        faces.push_back(f);
    }

    // Back-to-front så halvtransparent plasma blir snygg
    std::sort(faces.begin(), faces.end(),
        [](const Face& a, const Face& b) { return a.zavg > b.zavg; });

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    for (auto& f : faces) {
        SDL_Vertex v[6];
        auto setV = [](SDL_Vertex& vv, float x, float y, float u, float t, Uint8 a) {
            vv.position.x = x; vv.position.y = y;
            vv.color = SDL_Color{ 255,255,55,a };
            vv.tex_coord.x = u; vv.tex_coord.y = t;
            };

        // två trianglar per sida
        setV(v[0], (float)f.p0.x, (float)f.p0.y, f.u0, 0.f, 230);
        setV(v[1], (float)f.p1.x, (float)f.p1.y, f.u1, 0.f, 230);
        setV(v[2], (float)f.p2.x, (float)f.p2.y, f.u1, 1.f, 230);

        setV(v[3], (float)f.p0.x, (float)f.p0.y, f.u0, 0.f, 230);
        setV(v[4], (float)f.p2.x, (float)f.p2.y, f.u1, 1.f, 230);
        setV(v[5], (float)f.p3.x, (float)f.p3.y, f.u0, 1.f, 230);

        SDL_RenderGeometry(ren, tex, v, 6, nullptr, 0);
    }

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}

// --- C64 state ---
static GLuint C64Prog = 0, C64VAO = 0, C64VBO = 0, C64RandTex = 0;
static int C64GridW = 64, C64GridH = 36;
static int C64Reveal = 0;
static float C64Time = 0.f;
static float C64Decomp = 0.f;           // 0..1
static bool C64DecompPhase = true;      // visa decrunch först
static bool C64Done = false;
static bool C64Boot = false;
static float C64BootT = 0.f;
static float C64HoldT = 0.f;                     // time to hold pattern after reveal

// Text lines and timing for the boot sequence
static constexpr char C64CodeLine[] = "10 PRINT CHR$(205.5+RND(1));:GOTO 10";
static constexpr char C64RunLine[]  = "RUN";
static constexpr float C64TypeSpeed = 20.f;  // chars per second
static constexpr float C64BootDelay = 0.5f;  // wait before typing starts
static constexpr float C64RunDelay  = 0.3f;  // pause before RUN
static constexpr float C64AfterRunDelay = 0.5f; // pause after RUN before pattern
static constexpr float C64BootDuration =
    C64BootDelay + (sizeof(C64CodeLine)-1)/C64TypeSpeed +
    C64RunDelay  + (sizeof(C64RunLine)-1)/C64TypeSpeed +
    C64AfterRunDelay;
static constexpr float C64HoldDuration = 3.0f;   // show finished pattern

// Create GL_R8 random texture of size w x h
static GLuint makeRandTex(int w, int h) {
    std::vector<uint8_t> rnd(w * h);
    for (auto& v : rnd) v = uint8_t(rand() & 255);
    GLuint t = 0; glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, rnd.data());
    return t;
}

void initC64Window(int screenW, int screenH) {
    // program
    C64Prog = makeProgram(c64_vs, c64_fs);

    // fullscreen quad
    float quad[8] = { -1.f,-1.f,  1.f,-1.f,  -1.f,1.f,  1.f,1.f };
    glGenVertexArrays(1, &C64VAO);
    glGenBuffers(1, &C64VBO);
    glBindVertexArray(C64VAO);
    glBindBuffer(GL_ARRAY_BUFFER, C64VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // grid: lite "C64-ish" teckenupplösning
    C64GridW = 80;  // fler kolumner = tunnare diagonaler
    C64GridH = 50;
    C64RandTex = makeRandTex(C64GridW, C64GridH);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void startC64Window() { // resetta för nästa visning om du vill
    C64Reveal = 0;
    C64Time = 0.f;
    C64Decomp = 0.f;
    C64DecompPhase = true;
    C64Done = false;
    C64Boot = false;
    C64BootT = 0.f;
    C64HoldT = 0.f;
}

void updateC64Window(float dt) {
    C64Time += dt;

    if (C64DecompPhase) {

        const float decompDur = 2.0f;  // a couple of seconds of decrunch
        C64Decomp = std::min(1.f, C64Time / decompDur);
        if (C64Decomp >= 1.f) {
            C64DecompPhase = false;
            C64Boot = true;
            C64BootT = 0.f;
            C64Reveal = 0;
        }
    } else if (C64Boot) {
        C64BootT += dt;
        if (C64BootT > C64BootDuration) {
            C64Boot = false;
        }
    } else {
        int total = C64GridW * C64GridH;
        int add = int(1200.f * dt); // draw the pattern at a readable pace
        C64Reveal = std::min(total, C64Reveal + add);
        if (C64Reveal == total) {
            C64HoldT += dt;
            if (C64HoldT >= C64HoldDuration) {
                C64Done = true; // pattern shown long enough
            }
        }
    }
}

bool c64WindowIsDone() { return C64Done; }
void c64RequestFly() { /* valfritt: resetta för ny reveal */ startC64Window(); }

// Renderar allt i en pass: decrunch-bg + panel i mitten
void renderC64Window(int screenW, int screenH) {
    glUseProgram(C64Prog);

    // panelstorlek: 60% av höjd, och samma aspect som skärmen
    float panelH = 0.60f;                       // 60% av höjden
    float aspect = float(screenW) / float(screenH);
    float panelW = panelH * aspect;             // så rutan blir rektangulär, inte kvadratisk
    panelW = std::min(panelW, 0.85f);           // clamp – inte större än 85% bredd
    // vill du ha mindre: sänk panelH, t.ex. 0.50f

    // center i mitten
    float cx = 0.5f, cy = 0.5f;
    GLint loc;

    // uniforms
    if ((loc = glGetUniformLocation(C64Prog, "uRand")) != -1) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, C64RandTex);
        glUniform1i(loc, 0);
    }
    if ((loc = glGetUniformLocation(C64Prog, "uGrid")) != -1) glUniform2i(loc, C64GridW, C64GridH);
    if ((loc = glGetUniformLocation(C64Prog, "uRevealCount")) != -1) glUniform1i(loc, C64Reveal);
    if ((loc = glGetUniformLocation(C64Prog, "uThickness")) != -1) glUniform1f(loc, 0.06f);
    if ((loc = glGetUniformLocation(C64Prog, "uInk")) != -1)  glUniform3f(loc, 0.90f, 0.90f, 0.95f);
    if ((loc = glGetUniformLocation(C64Prog, "uBG")) != -1)   glUniform3f(loc, 0.09f, 0.20f, 0.85f);  // C64-blå
    if ((loc = glGetUniformLocation(C64Prog, "uEdge")) != -1) glUniform3f(loc, 0.05f, 0.10f, 0.40f);
    if ((loc = glGetUniformLocation(C64Prog, "uPanelRect")) != -1) glUniform4f(loc, cx, cy, panelW, panelH);
    if ((loc = glGetUniformLocation(C64Prog, "uTime")) != -1)  glUniform1f(loc, C64Time);
    if ((loc = glGetUniformLocation(C64Prog, "uDecomp")) != -1)glUniform1f(loc, C64DecompPhase ? C64Decomp : 1.0f);

    glBindVertexArray(C64VAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    if (C64Boot) {
        int w = int(panelW * screenW);
        int h = int(panelH * screenH);
        int x = (screenW - w) / 2;
        int y = (screenH - h) / 2;
        TTF_Font* f = consoleFont ? consoleFont : menuFont;
        SDL_Color col{120,255,120,255};
        auto drawLine = [&](const char* s, int line) {
            if (SDL_Texture* t = renderText(renderer, f, s, col)) {
                int tw, th; SDL_QueryTexture(t, nullptr, nullptr, &tw, &th);
                int dx = x + 20;
                int dy = y + 20 + line * (th + 4);
                glDrawSDLTextureOverlay(t, dx, dy, tw, th, screenW, screenH);
                SDL_DestroyTexture(t);
            }
        };

        float t = C64BootT;
        drawLine("**** COMMODORE 64 BASIC V2 ****", 0);
        drawLine("64K RAM SYSTEM  38911 BASIC BYTES FREE", 1);
        drawLine("READY.", 2);

        if (t > C64BootDelay) {
            int maxChars = (int)(sizeof(C64CodeLine) - 1);
            int chars = std::min<int>((t - C64BootDelay) * C64TypeSpeed, maxChars);
            std::string code(C64CodeLine, chars);
            drawLine(code.c_str(), 4);
        }
        float startRun = C64BootDelay + (sizeof(C64CodeLine)-1)/C64TypeSpeed + C64RunDelay;
        if (t > startRun) {
            int maxChars = (int)(sizeof(C64RunLine) - 1);
            int chars = std::min<int>((t - startRun) * C64TypeSpeed, maxChars);
            std::string runStr(C64RunLine, chars);
            drawLine(runStr.c_str(), 5);
        }
    }
}


int main(int argc, char* argv[]) {
    srand((unsigned)time(nullptr));

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << "\n"; return 1;
    }
    if (TTF_Init() == -1) {
        std::cerr << "TTF_Init Error: " << TTF_GetError() << "\n"; SDL_Quit(); return 1;
    }
    if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0) {
        std::cerr << "Mix_OpenAudio Error: " << Mix_GetError() << "\n";
    }

    gARGB = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);
    backgroundMusic = Mix_LoadMUS("backgroundmusic.wav");
    demoMusic = Mix_LoadMUS("musik1.wav");
    if (backgroundMusic) {
        Mix_PlayMusic(backgroundMusic, -1);
        currentMusic = backgroundMusic;
    }

    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");

    // --- Skapa fönster ---
    SDL_Window* window = SDL_CreateWindow("title",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        SCREEN_WIDTH, SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    if (!window) { std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << "\n"; return 1; }

    // --- Skapa renderer (exakt en gång) ---
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) { std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << "\n"; return 1; }

    // --- Skapa GL-context (exakt en gång) ---
    SDL_GLContext glctx = SDL_GL_CreateContext(window);
    if (!glctx) {
        std::cerr << "SDL_GL_CreateContext Error: " << SDL_GetError() << "\n";
        return 1;
    }

    // Koppla till globala pekarna du definierade högst upp i filen
    gWindow = window;
    gGL = glctx;

    // SDL_GL_SetSwapInterval(1); // Enable VSync
    // glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW initialization failed\n";
    }

    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    float f = 1.f / tanf(45.f * PI / 360.f);
    float zNear = 1.f, zFar = 100.f;
    float proj[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zFar + zNear) / (zNear - zFar), -1,
        0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
    };
    glLoadMatrixf(proj);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat amb[] = { 0.2f,0.2f,0.2f,1.f };
    GLfloat diff[] = { 0.8f,0.8f,0.8f,1.f };
    GLfloat pos[] = { 0.f,0.f,2.f,1.f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diff);
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    glEnable(GL_MULTISAMPLE);

    // === C64 10 PRINT === init (GL är redo, SCREEN_* satta)
    initC64Window(SCREEN_WIDTH, SCREEN_HEIGHT);

    IMG_Init(IMG_INIT_PNG);
    logoTexture = IMG_LoadTexture(renderer, "logo.png");
    if (logoTexture) SDL_QueryTexture(logoTexture, nullptr, nullptr, &logoWidth, &logoHeight);

    menuFont = TTF_OpenFont("Roboto-VariableFont_wdth,wght.ttf", 50);
    consoleFont = TTF_OpenFont("Roboto-VariableFont_wdth,wght.ttf", 30);
    bigFont = TTF_OpenFont("Roboto-VariableFont_wdth,wght.ttf", 96);
    titleFont = TTF_OpenFont("Roboto-VariableFont_wdth,wght.ttf", 25);
    scrollFont = TTF_OpenFont("Roboto-VariableFont_wdth,wght.ttf", 30);
    PongFont = TTF_OpenFont("Roboto-VariableFont_wdth,wght.ttf", 25);
    if (menuFont) TTF_SetFontStyle(menuFont, TTF_STYLE_BOLD);
    if (bigFont)  TTF_SetFontStyle(bigFont, TTF_STYLE_BOLD);
    hoverSound = Mix_LoadWAV("hover.wav");
    bangSound = Mix_LoadWAV("bang.wav");

    // Init “plasma”-textur och blandningsläge
    gPlasma = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, gPlasmaW, gPlasmaH);
    SDL_SetTextureBlendMode(gPlasma, SDL_BLENDMODE_BLEND);

    SDL_Color btnCol = { 200, 200, 200, 255 };
    backButtonTexture = renderText(renderer, menuFont, "Back", btnCol);
    nextButtonTexture = renderText(renderer, menuFont, "Next", btnCol);
    quitButtonTexture = renderText(renderer, menuFont, "Quit", btnCol);
    int bw, bh; SDL_QueryTexture(backButtonTexture, nullptr, nullptr, &bw, &bh);
    int sw, sh; SDL_QueryTexture(nextButtonTexture, nullptr, nullptr, &sw, &sh);
    int qw, qh; SDL_QueryTexture(quitButtonTexture, nullptr, nullptr, &qw, &qh);
    int spacing = 30;
    int total = bw + qw + sw + spacing * 2;
    int startX = (SCREEN_WIDTH - total) / 2;
    int baseY = 150;
    backButtonRect = { startX, baseY, bw, bh };
    quitButtonRect = { startX + bw + spacing, baseY, qw, qh };
    nextButtonRect = { startX + bw + spacing + qw + spacing, baseY, sw, sh };
    initMenu();
    initPortfolio();

    bool running = true;
    Uint32 lastTicks = SDL_GetTicks();

    while (running) {
        Uint32 cur = SDL_GetTicks();
        bool usedGLThisFrame = false;
        float deltaTime = (cur - lastTicks) / 1000.0f;
        lastTicks = cur;
        gLastDt = deltaTime;

        elapsedTime += deltaTime;
        cubeAngleX += deltaTime * 2.0f;
        cubeAngleY += deltaTime * 1.2f;
        if (!starTransition) effectTimer += deltaTime;

        if (starTransition) {
            starTransitionTime += deltaTime;
            float t = starTransitionTime / starTransitionDuration;
            if (t > 1.f) t = 1.f;
            float mult = 3.f + 10.f * sinf(t * PI);
            starSpeedMultiplier = starTransitionToMenu ? -mult : mult;
            starYaw = startYaw + (endYaw - startYaw) * t;
            starPitch = startPitch + (endPitch - startPitch) * t;

            if (starTransitionTime >= starTransitionDuration) {
                starTransition = false;
                starSpeedMultiplier = 1.5f;
                starYaw = endYaw;
                starPitch = endPitch;
                pongGameOver = false;

                if (starTransitionToMenu) {
                    starTransitionToMenu = false;
                    currentState = STATE_MENU;
                    portfolioMusicStarted = false;
                    startBackgroundMusic();
                }
                else {
                    currentEffectIndex = targetEffectIndex;
                    startPortfolioEffect(effectSequence[currentEffectIndex]);
                }
            }
        }
        else {
            starSpeedMultiplier = 4.5f;
        }

        updateInterference(deltaTime);

        SDL_Event e;
        int mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);
        bool mouseClick = false;
        click = false;

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;

            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
                if (currentState == STATE_PORTFOLIO && !starTransition) {
                    startStarTransition(currentEffectIndex, true);
                }
                else if (currentState == STATE_ABOUT && !starTransition) {
                    startStarTransition(currentEffectIndex, true);
                }
                else if (currentState == STATE_MENU) {
                    running = false;
                }
            }
            else if (e.type == SDL_KEYDOWN &&
                currentState == STATE_PORTFOLIO &&
                currentPortfolioSubState == VIEW_SNAKE_GAME) {
                switch (e.key.keysym.sym) {
                case SDLK_RIGHT: snakeDir = 0; break;
                case SDLK_DOWN:  snakeDir = 1; break;
                case SDLK_LEFT:  snakeDir = 2; break;
                case SDLK_UP:    snakeDir = 3; break;
                default: break;
                }
            }
            else if (e.type == SDL_KEYDOWN &&
                currentState == STATE_PORTFOLIO &&
                currentPortfolioSubState == VIEW_PONG_GAME) {

                if (pongGameOver) {
                    if (e.key.keysym.sym == SDLK_RETURN ||
                        e.key.keysym.sym == SDLK_SPACE) {
                        startStarTransition(currentEffectIndex, true);
                    }
                    else if (e.key.keysym.sym == SDLK_r) {
                        initPongGame(); // restart only when finished
                    }
                }
                else {
                    if (e.key.keysym.sym == SDLK_UP)       paddleDir = -1;
                    else if (e.key.keysym.sym == SDLK_DOWN) paddleDir = 1;
                }
            }

            if (e.type == SDL_MOUSEBUTTONDOWN) { mouseClick = true; click = true; }
        }

        if (exitExplosion) {
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 0);
            SDL_Rect full = { 0,0,SCREEN_WIDTH,SCREEN_HEIGHT };
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            SDL_RenderFlush(renderer);
            renderFireworks(renderer, deltaTime);
            SDL_RenderPresent(renderer);

            exitTimer += deltaTime;
            if (exitTimer > 3.f && fireworks.empty()) {
                exitExplosion = false;
                if (explosionAdvanceEffect) {
                    explosionAdvanceEffect = false;
                    startStarTransition(explosionNextIndex);
                }
                else if (explosionReturnToMenu) {
                    currentState = STATE_MENU;
                    startBackgroundMusic();
                    explosionReturnToMenu = false;
                }
                else {
                    running = false;
                }
            }
            continue;
        }

        if (currentState == STATE_MENU) {

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            SDL_SetRenderDrawColor(renderer, 10, 10, 30, 255);
            SDL_RenderClear(renderer);
            renderLogoWithReflection(renderer, logoTexture, 300);
            renderStars(renderer, smallStars, { 180,180,255,220 });
            renderStars(renderer, bigStars, { 255,255,255,255 });
            renderStaticStars(renderer);
            renderAboutC64Typewriter(renderer, deltaTime);

            renderMenu(deltaTime, mouseX, mouseY, mouseClick);

            SDL_Texture* title = renderText(renderer,
                titleFont ? titleFont : menuFont,
                "(C)opyright 2025 Schy-DevelopmentTM   Email: matsarlemark@gmail.com   Version: 2025.01.06.11", { 200,200,200,120 });
            if (title) {
                int tw, th; SDL_QueryTexture(title, nullptr, nullptr, &tw, &th);
                SDL_Rect dst = { (SCREEN_WIDTH - tw) / 2, SCREEN_HEIGHT - th - 40, tw, th };
                SDL_RenderCopy(renderer, title, nullptr, &dst);
                SDL_DestroyTexture(title);
            }
        }
        else if (currentState == STATE_ABOUT) {
            SDL_SetRenderDrawColor(renderer, 10, 10, 30, 255);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderStars(renderer, smallStars, { 180,180,255,220 });
            renderStars(renderer, bigStars, { 255,255,255,255 });
            renderStaticStars(renderer);

            SDL_Point mp{ mouseX, mouseY };
            bool hov = SDL_PointInRect(&mp, &backButtonRect);
            if (hov && !backWasHovered && hoverSound) {
                Mix_PlayChannel(-1, hoverSound, 0);
                backWasHovered = true;
            }
            else if (!hov) {
                backWasHovered = false;
            }
            backHoverAnim += (hov ? 1.f : -1.f) * deltaTime * 6.f;
            backHoverAnim = clampValue(backHoverAnim, 0.f, 1.f);
            SDL_Color bc = hov ? SDL_Color{ 255,180,100,255 } : SDL_Color{ 200,200,200,255 };
            SDL_Texture* btnTex = renderText(renderer, menuFont, "Back", bc);
            SDL_Rect bDst = backButtonRect;
            bDst.y -= static_cast<int>(6.f * backHoverAnim);   // cast
            SDL_RenderCopy(renderer, btnTex, nullptr, &bDst);
            SDL_DestroyTexture(btnTex);

            if (mouseClick && hov && !starTransition) {
                startStarTransition(currentEffectIndex, true);
            }
        }
        else if (currentState == STATE_PORTFOLIO) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            SDL_SetRenderDrawColor(renderer, 10, 10, 10, 0);
            SDL_RenderClear(renderer);

            renderStars(renderer, smallStars, { 180,180,255,255 });
            renderStars(renderer, bigStars, { 255,255,255,255 });
            renderStaticStars(renderer);

            SDL_RenderFlush(renderer);

            if (!starTransition) {
                usedGLThisFrame = renderPortfolioEffect(renderer, deltaTime);
            }

            if (currentPortfolioSubState != VIEW_C64_10PRINT) {
                SDL_Point mp{ mouseX, mouseY };
                bool hovBack = SDL_PointInRect(&mp, &backButtonRect);
                bool hovNext = SDL_PointInRect(&mp, &nextButtonRect);
                bool hovQuit = SDL_PointInRect(&mp, &quitButtonRect);

                if (hovBack && !backWasHovered && hoverSound) {
                    Mix_PlayChannel(-1, hoverSound, 0);
                    backWasHovered = true;
                } else if (!hovBack) {
                    backWasHovered = false;
                }

                if (hovNext && !nextWasHovered && hoverSound) {
                    Mix_PlayChannel(-1, hoverSound, 0);
                    nextWasHovered = true;
                } else if (!hovNext) {
                    nextWasHovered = false;
                }

                if (hovQuit && !quitWasHovered && hoverSound) {
                    Mix_PlayChannel(-1, hoverSound, 0);
                    quitWasHovered = true;
                } else if (!hovQuit) {
                    quitWasHovered = false;
                }

                backHoverAnim += (hovBack ? 1.f : -1.f) * deltaTime * 6.f;
                backHoverAnim = clampValue(backHoverAnim, 0.f, 1.f);
                nextHoverAnim += (hovNext ? 1.f : -1.f) * deltaTime * 6.f;
                nextHoverAnim = clampValue(nextHoverAnim, 0.f, 1.f);
                quitHoverAnim += (hovQuit ? 1.f : -1.f) * deltaTime * 6.f;
                quitHoverAnim = clampValue(quitHoverAnim, 0.f, 1.f);

                SDL_Color bc = hovBack ? kMonoGreenHover : kMonoGreenBase;
                SDL_Color sc = hovNext ? kMonoGreenHover : kMonoGreenBase;
                SDL_Color qc = hovQuit ? kMonoGreenHover : kMonoGreenBase;

                SDL_Texture* backTex = renderText(renderer, menuFont, "[Back]", bc);
                SDL_Texture* nextTex = renderText(renderer, menuFont, "[Next]", sc);
                SDL_Texture* quitTex = renderText(renderer, menuFont, "[Main]", qc);

                SDL_Rect bdst = backButtonRect; bdst.y -= static_cast<int>(6.f * backHoverAnim);
                SDL_Rect ndst = nextButtonRect; ndst.y -= static_cast<int>(6.f * nextHoverAnim);
                SDL_Rect qdst = quitButtonRect; qdst.y -= static_cast<int>(6.f * quitHoverAnim);

                SDL_RenderCopy(renderer, backTex, nullptr, &bdst);
                SDL_RenderCopy(renderer, nextTex, nullptr, &ndst);
                SDL_RenderCopy(renderer, quitTex, nullptr, &qdst);

                applyDotMask(renderer, bdst);
                applyScanlines(renderer, bdst, 2);
                applyDotMask(renderer, ndst);
                applyScanlines(renderer, ndst, 2);
                applyDotMask(renderer, qdst);
                applyScanlines(renderer, qdst, 2);

                SDL_DestroyTexture(backTex);
                SDL_DestroyTexture(nextTex);
                SDL_DestroyTexture(quitTex);

                if (mouseClick && hovBack && !starTransition) {
                    int idx = (currentEffectIndex - 1 + NUM_EFFECTS) % NUM_EFFECTS;
                    startStarTransition(idx);
                } else if (mouseClick && hovNext && !starTransition) {
                    if (currentPortfolioSubState == VIEW_WIREFRAME_CUBE) {
                        if (!WF_FlyOut) {
                            WF_FlyOut = true;
                            WF_FlyT = 0.f;
                        }
                    } else {
                        int idx = (currentEffectIndex + 1) % NUM_EFFECTS;
                        startStarTransition(idx);
                    }
                } else if (mouseClick && hovQuit && !starTransition) {
                    startStarTransition(currentEffectIndex, true);
                }
            }
        }

        SDL_RenderFlush(renderer);
        if (usedGLThisFrame) {
            ensureGLContextCurrent();
            glFlush();
            SDL_GL_SwapWindow(window);
        }
        else {
            SDL_RenderPresent(renderer);
        }
    }

    Mix_HaltMusic();
    if (backgroundMusic) Mix_FreeMusic(backgroundMusic);
    if (demoMusic) Mix_FreeMusic(demoMusic);
    Mix_FreeChunk(hoverSound);
    Mix_FreeChunk(bangSound);
    Mix_CloseAudio();

    TTF_CloseFont(menuFont);
    TTF_CloseFont(PongFont);
    TTF_CloseFont(consoleFont);
    TTF_CloseFont(bigFont);
    TTF_CloseFont(titleFont);
    TTF_CloseFont(scrollFont);
    TTF_Quit();

    SDL_DestroyTexture(logoTexture);
    SDL_DestroyTexture(backButtonTexture);
    SDL_DestroyTexture(nextButtonTexture);
    SDL_DestroyTexture(quitButtonTexture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    IMG_Quit();
    if (gARGB) { SDL_FreeFormat(gARGB); gARGB = nullptr; }
    SDL_Quit();

    return 0;
}
