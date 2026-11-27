// Copy this file to secrets.h (same folder) and fill in YOUR WiFi network.
// secrets.h is git-ignored, so your password never reaches GitHub.
//
// Without secrets.h the firmware uses Wokwi's virtual open network "Wokwi-GUEST", which only
// exists in the Wokwi simulator (VS Code extension, wokwi-cli, GitHub Actions).
// The ESP32 only supports 2.4 GHz WiFi; a phone hotspot works well for demos.
#pragma once

#define TINYMLOPS_WIFI_SSID "your-2.4GHz-network"
#define TINYMLOPS_WIFI_PASSWORD "your-password"

// Optional: point the device at another manifest (e.g. a fork of the repository).
// #define TINYMLOPS_MANIFEST_URL "https://github.com/<owner>/<repo>/releases/latest/download/manifest.json"
