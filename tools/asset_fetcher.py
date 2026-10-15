"""标准国旗、国徽与地方旗帜 SVG 资源自动获取与解析管线。
支持：
1. ISO 3166-1 (国旗) 与 ISO 3166-2 (地方/省州旗帜，2800+面) 的高速检索与下载
2. Wikimedia Commons API 自动化抓取各国官方矢量国徽 (Coats of Arms)
3. SVG 矢量解析与图层解耦（提取底板、提取徽章、颜色篡改、拼接融合）
"""
import os
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_SVG_DIR = ROOT_DIR / "assets" / "raw_svg"
EMBLEMS_DIR = ROOT_DIR / "assets" / "emblems"
REGIONAL_DIR = ROOT_DIR / "assets" / "regional"


def ensure_dirs():
    for d in [RAW_SVG_DIR, EMBLEMS_DIR, REGIONAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


class StandardAssetFetcher:
    """标准旗帜与徽章资产采集引擎"""

    # 1. 权威国旗库 (ISO 3166-1 alpha-2, lipis/flag-icons)
    GITHUB_FLAG_ICONS_BASE = "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3"

    # 2. 全球最全地方旗帜库 (ISO 3166-2, amckenna41/iso3166-flags, 包含2800+州/省/区旗)
    GITHUB_ISO3166_2_BASE = "https://raw.githubusercontent.com/amckenna41/iso3166-flags/main/flags/svg"

    # 3. 维基共享资源 MediaWiki API (各国国徽官方矢量主干)
    WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

    @classmethod
    def download_country_flag(cls, iso_code: str, save_dir: Path = RAW_SVG_DIR) -> Optional[Path]:
        """下载标准国家旗帜 SVG（如 cn, us, de, fr, jp）"""
        ensure_dirs()
        iso = iso_code.lower()
        url = f"{cls.GITHUB_FLAG_ICONS_BASE}/{iso}.svg"
        target = save_dir / f"{iso}.svg"
        if target.exists():
            return target
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FlagFlywheelBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp, open(target, "wb") as f:
                f.write(resp.read())
            print(f"[fetcher] 成功获取国旗: {iso}.svg")
            return target
        except Exception as e:
            print(f"[fetcher] 获取国旗失败 {iso}: {e}")
            return None

    @classmethod
    def download_regional_flag(cls, iso_3166_2: str, save_dir: Path = REGIONAL_DIR) -> Optional[Path]:
        """下载地方/省州旗帜 SVG（例如 US-CA 加州, JP-13 东京都, DE-BY 巴伐利亚州）"""
        ensure_dirs()
        code = iso_3166_2.upper()
        url = f"{cls.GITHUB_ISO3166_2_BASE}/{code}.svg"
        target = save_dir / f"{code}.svg"
        if target.exists():
            return target
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FlagFlywheelBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp, open(target, "wb") as f:
                f.write(resp.read())
            print(f"[fetcher] 成功获取地方旗帜: {code}.svg")
            return target
        except Exception as e:
            print(f"[fetcher] 获取地方旗帜失败 {code}: {e}")
            return None

    @classmethod
    def fetch_national_emblem_from_wikimedia(cls, country_name_en: str, save_dir: Path = EMBLEMS_DIR) -> Optional[Path]:
        """从 Wikimedia Commons API 检索并下载各国官方国徽矢量 SVG。"""
        ensure_dirs()
        target_file = save_dir / f"emblem_{country_name_en.lower().replace(' ', '_')}.svg"
        if target_file.exists():
            return target_file

        title = f"File:Coat of arms of {country_name_en}.svg"
        params = {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        api_url = f"{cls.WIKIMEDIA_API}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "FlagFlywheelBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid == "-1":
                    print(f"[fetcher] 维基未检索到官方国徽文件: {title}")
                    return None
                img_url = pdata.get("imageinfo", [{}])[0].get("url")
                if img_url:
                    img_req = urllib.request.Request(img_url, headers={"User-Agent": "FlagFlywheelBot/1.0"})
                    with urllib.request.urlopen(img_req, timeout=15) as img_resp, open(target_file, "wb") as f:
                        f.write(img_resp.read())
                    print(f"[fetcher] 成功从维基获取国徽: {country_name_en} -> {target_file.name}")
                    return target_file
        except Exception as e:
            print(f"[fetcher] 维基国徽获取异常 {country_name_en}: {e}")
            return None


if __name__ == "__main__":
    # 测试拉取几面代表性国旗与地方旗帜
    print(">>> 启动资产采集测试...")
    StandardAssetFetcher.download_country_flag("cn")
    StandardAssetFetcher.download_country_flag("us")
    StandardAssetFetcher.download_regional_flag("US-CA")
    print(">>> 资产采集模块运行完毕。")
