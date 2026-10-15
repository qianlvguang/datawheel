"""权威国徽、纹章、航天/体育/文化勋章与全球地方行政区旗批量拉取器 (Full Emblems, Badges & Regional Flags Downloader)。

数据来源：
1. 维基百科与维基共享资源 (Armorial of Sovereign States) -> 全球所有 200+ 主权国家与权威实体独立高精国徽 (SVG)
2. 全球主要大国地方行政区旗 (加拿大省区、德国联邦州、西班牙自治区、日本都道府县、澳大利亚州区、英国组成国与属地、法国大区、意大利大区、瑞士联邦州、中国特别行政区)
3. 权威航天、国际组织与顶级体育文化徽章 (NASA、阿波罗、阿尔忒弥斯、联合国、奥运五环、红十字、欧冠顶级俱乐部徽章)

遵循原则：
- 严格标准矢量格式 (SVG)
- 礼貌限速与真实浏览器标头，稳定防 429
- 路径与命名规范统一
"""
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
EMBLEMS_DIR = ASSETS_DIR / "official_emblems"
REGIONAL_DIR = ASSETS_DIR / "official_regional"

EMBLEMS_DIR.mkdir(parents=True, exist_ok=True)
REGIONAL_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def safe_download(url: str, dst: Path, min_bytes: int = 150, delay: float = 0.08) -> bool:
    """安全下载单个 SVG 矢量文件，带存在性检查与重试机制。"""
    if dst.exists() and dst.stat().st_size >= min_bytes:
        return True

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                # 验证 SVG 合法性
                if b"<svg" in content.lower():
                    with open(dst, "wb") as f:
                        f.write(content)
                    time.sleep(delay)
                    return True
                else:
                    return False
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 * (attempt + 1))
            else:
                time.sleep(delay)
                return False
        except Exception:
            time.sleep(delay)
            return False
    return False


def fetch_page_svgs(page_url: str):
    """从维基百科页面解析所有 SVG 资源哈希路径与原始文件名。"""
    try:
        req = urllib.request.Request(page_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8")
        pattern = r'thumb\.wikimedia\.org/wikipedia/commons/thumb/([0-9a-f]/[0-9a-f]{2})/([^/\"\'\s]+?\.svg)/\d+px-'
        matches = re.findall(pattern, html)
        seen = set()
        items = []
        for h, raw_name in matches:
            if raw_name not in seen:
                seen.add(raw_name)
                items.append((h, raw_name))
        return items
    except Exception as e:
        print(f"  ✗ 抓取页面失败 {page_url}: {e}")
        return []


def download_all_sovereign_emblems():
    """1. 全球主权国家与独立实体国徽/官方纹章全量入库 (~200+ 国家)。"""
    print("\n" + "=" * 80)
    print(" 🛡️ [权威国徽与官方纹章全量扩展] 正在从 Armorial of sovereign states 爬取...")
    print("=" * 80)

    url = "https://en.wikipedia.org/wiki/Armorial_of_sovereign_states"
    items = fetch_page_svgs(url)
    print(f"  ▶ 发现主权国家与实体国徽条目: {len(items)} 个，启动全量下载...")

    success = 0
    for h, raw_name in items:
        # 解码并规范命名
        decoded_name = urllib.parse.unquote(raw_name)
        clean_name = re.sub(r"^(?:Coat_of_arms_of_|National_Emblem_of_|Emblem_of_|Arms_of_|Seal_of_|Great_Seal_of_)", "", decoded_name, flags=re.I)
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name).lower()
        if not clean_name.endswith(".svg"):
            clean_name += ".svg"
        dst = EMBLEMS_DIR / f"emblem_{clean_name}"

        # 原始维基资源真实直链
        source_url = f"https://upload.wikimedia.org/wikipedia/commons/{h}/{raw_name}"
        if safe_download(source_url, dst):
            success += 1
            if success % 20 == 0 or success <= 5:
                print(f"  ✓ [{success}/{len(items)}] 国徽入库: {dst.name} ({dst.stat().st_size} bytes)")

    print(f"  ✨ 全球权威国徽与主权纹章全量完成: 成功入库 {success} 款！")


def download_all_world_regional_flags():
    """2. 全球主要国家下辖行政区旗全量扩展 (加拿大、德国、西班牙、日本、澳洲、英国、法国、意大利、瑞士、中国港澳)。"""
    print("\n" + "=" * 80)
    print(" 🚩 [全球地方省州行政区旗全量扩展] 启动加拿大、德国、西班牙、日本、英国等区旗...")
    print("=" * 80)

    sources = [
        # (国家代号, 维基页面, 过滤规则, 名称清洗)
        (
            "ca",
            "https://en.wikipedia.org/wiki/List_of_Canadian_flags",
            lambda n: "flag_of_" in n.lower() and any(p in n.lower() for p in [
                "ontario", "quebec", "nova_scotia", "new_brunswick", "manitoba", "british_columbia",
                "prince_edward", "saskatchewan", "alberta", "newfoundland", "northwest_territories", "yukon", "nunavut"
            ]),
            lambda n: "ca_" + re.sub(r"^flag_of_(?:the_)?", "", n, flags=re.I)
        ),
        (
            "de",
            "https://en.wikipedia.org/wiki/Flags_of_German_states",
            lambda n: "flag_of_" in n.lower() and "map" not in n.lower() and "(" not in n,
            lambda n: "de_" + re.sub(r"^flag_of_", "", n, flags=re.I)
        ),
        (
            "es",
            "https://en.wikipedia.org/wiki/Flags_of_the_autonomous_communities_of_Spain",
            lambda n: "flag_of_" in n.lower() and "map" not in n.lower(),
            lambda n: "es_" + re.sub(r"^flag_of_(?:the_)?", "", n, flags=re.I)
        ),
        (
            "jp",
            "https://en.wikipedia.org/wiki/List_of_Japanese_flags",
            lambda n: "prefecture" in n.lower() or any(p in n.lower() for p in ["tokyo", "kyoto", "osaka", "hokkaido"]),
            lambda n: "jp_" + re.sub(r"^flag_of_", "", n, flags=re.I)
        ),
        (
            "au",
            "https://en.wikipedia.org/wiki/List_of_Australian_flags",
            lambda n: "flag_of_" in n.lower() and any(p in n.lower() for p in [
                "new_south_wales", "victoria", "queensland", "western_australia", "south_australia", "tasmania",
                "australian_capital_territory", "northern_territory", "norfolk_island"
            ]) and "governor" not in n.lower() and "(" not in n,
            lambda n: "au_" + re.sub(r"^flag_of_(?:the_)?", "", n, flags=re.I)
        ),
        (
            "gb",
            "https://en.wikipedia.org/wiki/List_of_British_flags",
            lambda n: "flag_of_" in n.lower() and any(p in n.lower() for p in [
                "scotland", "wales", "england", "northern_ireland", "guernsey", "jersey", "isle_of_man",
                "gibraltar", "bermuda", "falkland", "cayman"
            ]) and "governor" not in n.lower() and "(" not in n and "parish" not in n.lower(),
            lambda n: "gb_" + re.sub(r"^flag_of_(?:the_)?", "", n, flags=re.I)
        ),
        (
            "it",
            "https://en.wikipedia.org/wiki/Flags_of_the_regions_of_Italy",
            lambda n: "flag_of_" in n.lower() and "map" not in n.lower() and "(" not in n,
            lambda n: "it_" + re.sub(r"^flag_of_", "", n, flags=re.I)
        ),
        (
            "fr",
            "https://en.wikipedia.org/wiki/Flags_of_the_regions_of_France",
            lambda n: "flag_of_" in n.lower() and ("region" in n.lower() or any(p in n.lower() for p in ["corsica", "brittany", "normandy"])) and "(" not in n,
            lambda n: "fr_" + re.sub(r"^flag_of_(?:the_)?(?:region_of_)?", "", n, flags=re.I)
        ),
        (
            "ch",
            "https://en.wikipedia.org/wiki/Flags_and_arms_of_cantons_of_Switzerland",
            lambda n: ("flag" in n.lower() or "che_" in n.lower()) and "map" not in n.lower() and "(" not in n,
            lambda n: "ch_" + re.sub(r"^(?:flag_of_canton_of_|che_)?", "", n, flags=re.I)
        ),
    ]

    total_downloaded = 0
    for country_code, page_url, filter_fn, name_fn in sources:
        items = fetch_page_svgs(page_url)
        filtered = [(h, name) for h, name in items if filter_fn(name)]
        print(f"  ▶ [{country_code.upper()}] 匹配到地方行政旗帜: {len(filtered)} 款...")
        cnt = 0
        for h, raw_name in filtered:
            decoded_name = urllib.parse.unquote(raw_name)
            clean_name = name_fn(decoded_name)
            clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name).lower()
            if not clean_name.endswith(".svg"):
                clean_name += ".svg"
            dst = REGIONAL_DIR / clean_name

            source_url = f"https://upload.wikimedia.org/wikipedia/commons/{h}/{raw_name}"
            if safe_download(source_url, dst):
                cnt += 1
                total_downloaded += 1
        print(f"    ✓ {country_code.upper()} 地方旗帜已入库: {cnt} 款")

    # 3. 补充中国特别行政区 (港澳) 旗帜
    sar_flags = [
        ("cn_hong_kong.svg", "https://upload.wikimedia.org/wikipedia/commons/5/5b/Flag_of_Hong_Kong.svg"),
        ("cn_macau.svg", "https://upload.wikimedia.org/wikipedia/commons/6/63/Flag_of_Macau.svg"),
    ]
    for filename, url in sar_flags:
        dst = REGIONAL_DIR / filename
        if safe_download(url, dst):
            total_downloaded += 1
            print(f"    ✓ 中国特别行政区旗帜入库: {filename}")

    print(f"  ✨ 全球地方行政区旗扩展完成: 本轮共计入库 {total_downloaded} 款！")


def download_iconic_badges_and_crests():
    """3. 权威航天任务、国际组织、顶级体育俱乐部与高校徽章入库。"""
    print("\n" + "=" * 80)
    print(" 🎖️ [航天/国际组织/顶级体育文化勋章入库] 正在拉取经典权威徽章...")
    print("=" * 80)

    badges = [
        # 航天与科学徽章 (Space & Science Badges)
        ("badge_nasa_meatball_insignia.svg", "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg"),
        ("badge_nasa_worm_logo.svg", "https://upload.wikimedia.org/wikipedia/commons/b/ba/NASA_Worm_logo.svg"),
        ("badge_apollo_11_mission.svg", "https://upload.wikimedia.org/wikipedia/commons/2/27/Apollo_11_insignia.svg"),
        ("badge_artemis_program.svg", "https://upload.wikimedia.org/wikipedia/commons/5/58/NASA_Artemis_logo.svg"),
        ("badge_esa_european_space_agency.svg", "https://upload.wikimedia.org/wikipedia/commons/6/61/ESA_logo_2020.svg"),
        
        # 国际组织与公约徽章 (International Organizations)
        ("badge_united_nations.svg", "https://upload.wikimedia.org/wikipedia/commons/2/2f/Flag_of_the_United_Nations.svg"),
        ("badge_olympic_rings.svg", "https://upload.wikimedia.org/wikipedia/commons/5/5c/Olympic_rings_without_rims.svg"),
        ("badge_red_cross.svg", "https://upload.wikimedia.org/wikipedia/commons/5/52/Flag_of_the_Red_Cross.svg"),
        ("badge_interpol_emblem.svg", "https://upload.wikimedia.org/wikipedia/commons/9/91/Emblem_of_Interpol.svg"),
        ("badge_nato_emblem.svg", "https://upload.wikimedia.org/wikipedia/commons/3/37/Flag_of_NATO.svg"),

        # 顶级体育与足球俱乐部徽章 (World Football & Sports Crests - 常见服装印花标的)
        ("crest_real_madrid_cf.svg", "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg"),
        ("crest_fc_barcelona.svg", "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg"),
        ("crest_manchester_united.svg", "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg"),
        ("crest_liverpool_fc.svg", "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg"),
        ("crest_fc_bayern_munich.svg", "https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg"),
        ("crest_arsenal_fc.svg", "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg"),
        ("crest_chelsea_fc.svg", "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg"),
        ("crest_ac_milan.svg", "https://upload.wikimedia.org/wikipedia/commons/d/d0/A.C._Milan_logo.svg"),
        ("crest_inter_milan.svg", "https://upload.wikimedia.org/wikipedia/commons/0/05/FC_Internazionale_Milano_2021.svg"),
        ("crest_juventus_fc.svg", "https://upload.wikimedia.org/wikipedia/commons/d/d2/Juventus_Club_Logo_2017.svg"),
        ("crest_borussia_dortmund.svg", "https://upload.wikimedia.org/wikipedia/commons/6/67/Borussia_Dortmund_logo.svg"),
        ("crest_paris_saint_germain.svg", "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg"),

        # 顶级高校与学术盾徽 (Academic Crests / Shields)
        ("crest_harvard_shield.svg", "https://upload.wikimedia.org/wikipedia/en/2/29/Harvard_shield_w_motto.svg"),
        ("crest_oxford_university.svg", "https://upload.wikimedia.org/wikipedia/commons/2/2f/University_of_Oxford_coat_of_arms.svg"),
        ("crest_cambridge_university.svg", "https://upload.wikimedia.org/wikipedia/commons/c/c3/Coat_of_Arms_of_the_University_of_Cambridge.svg"),
        ("crest_mit_seal.svg", "https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_Seal.svg"),
        ("crest_stanford_seal.svg", "https://upload.wikimedia.org/wikipedia/commons/b/b5/Seal_of_Stanford_University.svg"),
    ]

    success = 0
    for filename, url in badges:
        dst = EMBLEMS_DIR / filename
        if safe_download(url, dst):
            success += 1
            print(f"  ✓ 勋章/徽章就绪: {filename} ({dst.stat().st_size} bytes)")
    print(f"  ✨ 经典航天/体育/高校徽章入库完成: 共 {success} 款！")


def run():
    download_all_sovereign_emblems()
    download_all_world_regional_flags()
    download_iconic_badges_and_crests()


if __name__ == "__main__":
    run()
