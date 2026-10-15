"""下载权威标准官方 SVG 国旗、地方旗帜与国徽。
来源：
- lipis/flag-icons (ISO 3166-1 标准国旗)
- amckenna41/iso3166-flags (ISO 3166-2 标准地方州/省旗)
- Wikimedia Commons (官方矢量国徽与纹章)
"""
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FLAGS_DIR = ASSETS_DIR / "official_flags"
REGIONAL_DIR = ASSETS_DIR / "official_regional"
EMBLEMS_DIR = ASSETS_DIR / "official_emblems"

for d in [FLAGS_DIR, REGIONAL_DIR, EMBLEMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. 经典标准国旗 SVG
OFFICIAL_FLAGS = {
    "cn": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/cn.svg",  # 中国 (五星)
    "us": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/us.svg",  # 美国 (50星+13条纹)
    "eu": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/eu.svg",  # 欧盟 (12环形金星)
    "vn": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/vn.svg",  # 越南 (中心单星)
    "so": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/so.svg",  # 索马里 (蓝底单星)
    "fr": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/fr.svg",  # 法国 (蓝白红竖三色)
    "de": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/de.svg",  # 德国 (黑红金横三色)
    "it": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/it.svg",  # 意大利 (绿白红竖三色)
    "pl": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/pl.svg",  # 波兰 (白红二分横条)
    "ua": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/ua.svg",  # 乌克兰 (蓝黄二分横条)
    "se": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/se.svg",  # 瑞典 (蓝黄北欧十字)
    "ch": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/ch.svg",  # 瑞士 (红底正十字)
    "br": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/br.svg",  # 巴西 (绿底黄菱形+天球星图+文字)
    "es": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/es.svg",  # 西班牙 (带国徽及文字)
    "mx": "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/mx.svg",  # 墨西哥 (带鹰蛇仙人掌国徽)
}

# 2. 代表性地方行政区划旗帜 (ISO 3166-2)
REGIONAL_FLAGS = {
    "us_ca": "https://raw.githubusercontent.com/amckenna41/iso3166-flags/main/flags/svg/US-CA.svg",  # 加州熊旗 + 单星 + 文字
    "us_tx": "https://raw.githubusercontent.com/amckenna41/iso3166-flags/main/flags/svg/US-TX.svg",  # 德州孤星旗
    "us_nm": "https://raw.githubusercontent.com/amckenna41/iso3166-flags/main/flags/svg/US-NM.svg",  # 新墨西哥州 Zia 太阳符号
}

# 3. 官方国徽独立矢量图 (Wikimedia Commons SVG)
OFFICIAL_EMBLEMS = {
    "de_eagle": "https://upload.wikimedia.org/wikipedia/commons/2/22/Coat_of_arms_of_Germany.svg",  # 德国联邦之鹰国徽
    "es_shield": "https://upload.wikimedia.org/wikipedia/commons/8/85/Escudo_de_Espa%C3%B1a_%28her%C3%A1ldico%29.svg",  # 西班牙皇家盾徽
}


def download_all():
    print("[fetcher] 开始下载官方标准国旗 SVG...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for code, url in OFFICIAL_FLAGS.items():
        dst = FLAGS_DIR / f"{code}.svg"
        if not dst.exists():
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as r, open(dst, "wb") as f:
                    f.write(r.read())
                print(f"  ✓ 国旗 {code}.svg 已下载")
            except Exception as e:
                print(f"  ✗ 下载国旗 {code} 失败: {e}")

    print("[fetcher] 开始下载官方标准地方旗帜 SVG...")
    for code, url in REGIONAL_FLAGS.items():
        dst = REGIONAL_DIR / f"{code}.svg"
        if not dst.exists():
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as r, open(dst, "wb") as f:
                    f.write(r.read())
                print(f"  ✓ 地方旗帜 {code}.svg 已下载")
            except Exception as e:
                print(f"  ✗ 下载地方旗帜 {code} 失败: {e}")

    print("[fetcher] 开始下载官方矢量国徽 SVG...")
    for code, url in OFFICIAL_EMBLEMS.items():
        dst = EMBLEMS_DIR / f"{code}.svg"
        if not dst.exists():
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as r, open(dst, "wb") as f:
                    f.write(r.read())
                print(f"  ✓ 国徽 {code}.svg 已下载")
            except Exception as e:
                print(f"  ✗ 下载国徽 {code} 失败: {e}")


if __name__ == "__main__":
    download_all()
