# PlatformIO extra script: after every build, write .pio/build/<env>/firmware.merged.bin, a full
# 4 MB flash image (bootloader @0x1000, partition table @0x8000, boot_app0 @0xe000, app @0x10000).
#
# Why: given only the app-only firmware.bin, Wokwi (VS Code + CLI) substitutes its own partition
# table, which has no data partition, so LittleFS (the OTA model slots) cannot mount. With the
# merged image, the simulator boots with OUR partition table, exactly like a real ESP32.
# (docs.wokwi.com/guides/esp32 and /vscode/platformio)
import os

Import("env")  # noqa: F821  (provided by PlatformIO/SCons)


def merge_bin(source, target, env):  # noqa: ARG001
    build = env.subst("$BUILD_DIR")
    fw_dir = env.PioPlatform().get_package_dir("framework-arduinoespressif32")
    esptool = os.path.join(env.PioPlatform().get_package_dir("tool-esptoolpy"), "esptool.py")
    parts = [
        ("0x1000", os.path.join(build, "bootloader.bin")),
        ("0x8000", os.path.join(build, "partitions.bin")),
        ("0xe000", os.path.join(fw_dir, "tools", "partitions", "boot_app0.bin")),
        ("0x10000", os.path.join(build, "firmware.bin")),
    ]
    out = os.path.join(build, "firmware.merged.bin")
    args = " ".join(f'{off} "{path}"' for off, path in parts)
    cmd = (f'"$PYTHONEXE" "{esptool}" --chip esp32 merge_bin --flash_mode dio --flash_freq 40m '
           f'--flash_size 4MB -o "{out}" {args}')
    if env.Execute(cmd) != 0:
        raise RuntimeError("esptool merge_bin failed")
    print(f"[merge_firmware] wrote {out}")


env.AddPostAction("$BUILD_DIR/${PROGNAME}.bin", merge_bin)  # noqa: F821
