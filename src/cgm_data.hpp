#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct CGMRecord {
    std::string patient_id;
    std::vector<int64_t> t_min;
    std::vector<double> glucose;
    std::vector<int64_t> bolus_t_min;
    std::vector<double> bolus_units;
    std::vector<int64_t> meal_t_min;
    std::vector<double> meal_carbs_g;
};

struct CGMWindow {
    std::string patient_id;
    int64_t t_anchor_min;
    std::vector<double> lookback;
    std::vector<double> tod_sin;
    std::vector<double> tod_cos;
    std::vector<double> iob;
    std::vector<double> cob;
    std::vector<double> future_glucose;
    double future_min_glucose;
    int label;
    bool in_post_bolus_window;
};

struct CGMDataset {
    int lookback_steps;
    int horizon_steps;
    int step_minutes;
    double hypo_threshold;
    int post_bolus_max_min;
    std::vector<CGMWindow> train;
    std::vector<CGMWindow> val;
    std::vector<CGMWindow> test;
};

std::vector<CGMRecord> load_cgm_csv(const std::string& path);

CGMDataset make_windows(const std::vector<CGMRecord>& records,
                        int lookback_steps,
                        int horizon_steps,
                        int step_minutes,
                        double hypo_threshold,
                        int post_bolus_max_min,
                        double val_frac,
                        double test_frac,
                        uint32_t seed,
                        int window_stride = 1);

void normalize_inplace(CGMDataset& ds, double& out_mean, double& out_std);
