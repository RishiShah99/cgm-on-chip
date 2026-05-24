#include <Arduino.h>
#include "osdn_inference.h"
#include "osdn_weights.h"

using osdn_inf::H_MAX;
using osdn_inf::K_MAX;

static osdn_inf::Weights g_weights;
static osdn_inf::State   g_state;

static float g_ring[osdn_weights::LOOKBACK];
static int   g_ring_head = 0;
static int   g_ring_count = 0;

static constexpr int LED_PIN = 48;
static constexpr float HYPO_LOGIT_THRESHOLD = 0.0f;

static float normalize(float mg_dl) {
    return (mg_dl - osdn_weights::MEAN_MG_DL) / osdn_weights::STD_MG_DL;
}

static void push_reading(float mg_dl) {
    g_ring[g_ring_head] = normalize(mg_dl);
    g_ring_head = (g_ring_head + 1) % osdn_weights::LOOKBACK;
    if (g_ring_count < osdn_weights::LOOKBACK) g_ring_count++;
}

static float run_inference() {
    static float sig[osdn_weights::LOOKBACK];
    int n = g_ring_count;
    int start = (g_ring_head - n + osdn_weights::LOOKBACK) % osdn_weights::LOOKBACK;
    for (int t = 0; t < n; ++t) sig[t] = g_ring[(start + t) % osdn_weights::LOOKBACK];
    return osdn_inf::forward(g_weights, g_state, sig, n);
}

void setup() {
    Serial.begin(115200);
    delay(500);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    bool ok = osdn_inf::load_blob(
        g_weights,
        osdn_weights::H,
        osdn_weights::K,
        osdn_weights::BLOB,
        osdn_weights::BLOB_LEN);

    Serial.printf("OSDN load: %s  H=%d K=%d L=%d  params=%u\n",
                  ok ? "ok" : "FAIL",
                  osdn_weights::H, osdn_weights::K,
                  osdn_weights::LOOKBACK,
                  static_cast<unsigned>(osdn_weights::BLOB_LEN));
    osdn_inf::reset(g_state, osdn_weights::K);
}

void loop() {
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        float mg_dl = line.toFloat();
        if (mg_dl > 20.0f && mg_dl < 600.0f) {
            push_reading(mg_dl);
            if (g_ring_count >= osdn_weights::LOOKBACK) {
                float logit = run_inference();
                float prob  = 1.0f / (1.0f + expf(-logit));
                bool alert = logit > HYPO_LOGIT_THRESHOLD;
                digitalWrite(LED_PIN, alert ? HIGH : LOW);
                Serial.printf("cgm=%.1f logit=%.4f prob=%.4f alert=%d\n",
                              mg_dl, logit, prob, alert ? 1 : 0);
            } else {
                Serial.printf("cgm=%.1f buffering %d/%d\n",
                              mg_dl, g_ring_count, osdn_weights::LOOKBACK);
            }
        }
    }
    delay(10);
}
