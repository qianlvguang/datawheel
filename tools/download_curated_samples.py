"""三大核心品类精选样张下载器与资产库校准总控 (Curated Samples Downloader & Asset Calibrator)。

精准下载各品类 100 张高品质独立图样：
1. 动漫角色 IP (Anime / Character IPs): 100 款经典动漫角色官方全彩高保真透明图 (PokéAPI Official Artwork)
2. 奢侈品品牌 (Luxury Brands): 100 款国际顶级高奢时装/腕表/珠宝/超跑品牌官方标准矢量 Logo (SVG)
3. IP 品牌 (Iconic Pop Culture & Gaming IPs): 100 款世界级游戏/影视/玩具/动漫 IP 品牌官方标准矢量 Logo (SVG)

执行后自动校准资产库结构与 manifests 结构化清单。
"""
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MANIFESTS_DIR = ASSETS_DIR / "manifests"

ANIME_DIR = ASSETS_DIR / "character_ips" / "anime_100"
LUXURY_DIR = ASSETS_DIR / "brand_logos" / "luxury_100"
IP_BRANDS_DIR = ASSETS_DIR / "brand_logos" / "ip_brands_100"
SIMPLE_ICONS_DIR = ASSETS_DIR / "brand_logos" / "icons"

for d in [ANIME_DIR, LUXURY_DIR, IP_BRANDS_DIR, MANIFESTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
}


def safe_download(url: str, dst: Path, min_bytes: int = 150, delay: float = 0.04) -> bool:
    """安全下载单个文件，存在则跳过；遇到 404 立即返回，不进行无效重试。"""
    if dst.exists() and dst.stat().st_size >= min_bytes:
        return True

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
                if len(data) >= min_bytes:
                    with open(dst, "wb") as f:
                        f.write(data)
                    time.sleep(delay)
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    return False


# =============================================================================
# 1. 动漫角色 IP (100 款经典角色官方全彩透明立绘)
# =============================================================================
POKEMON_100 = [
    (1, "bulbasaur", "妙蛙种子", "Grass/Poison"),
    (2, "ivysaur", "妙蛙草", "Grass/Poison"),
    (3, "venusaur", "妙蛙花", "Grass/Poison"),
    (4, "charmander", "小火龙", "Fire"),
    (5, "charmeleon", "火恐龙", "Fire"),
    (6, "charizard", "喷火龙", "Fire/Flying"),
    (7, "squirtle", "杰尼龟", "Water"),
    (8, "wartortle", "卡咪龟", "Water"),
    (9, "blastoise", "水箭龟", "Water"),
    (10, "caterpie", "绿毛虫", "Bug"),
    (11, "metapod", "铁甲蛹", "Bug"),
    (12, "butterfree", "巴大蝶", "Bug/Flying"),
    (13, "weedle", "独角虫", "Bug/Poison"),
    (14, "kakuna", "铁壳蛹", "Bug/Poison"),
    (15, "beedrill", "大针蜂", "Bug/Poison"),
    (16, "pidgey", "波波", "Normal/Flying"),
    (17, "pidgeotto", "比比鸟", "Normal/Flying"),
    (18, "pidgeot", "大比鸟", "Normal/Flying"),
    (19, "rattata", "小拉达", "Normal"),
    (20, "raticate", "拉达", "Normal"),
    (21, "spearow", "烈雀", "Normal/Flying"),
    (22, "fearow", "大嘴雀", "Normal/Flying"),
    (23, "ekans", "阿柏蛇", "Poison"),
    (24, "arbok", "阿柏怪", "Poison"),
    (25, "pikachu", "皮卡丘", "Electric"),
    (26, "raichu", "雷丘", "Electric"),
    (27, "sandshrew", "穿山鼠", "Ground"),
    (28, "sandslash", "穿山王", "Ground"),
    (29, "nidoran_f", "尼多兰", "Poison"),
    (30, "nidorina", "尼多娜", "Poison"),
    (31, "nidoqueen", "尼多后", "Poison/Ground"),
    (32, "nidoran_m", "尼多朗", "Poison"),
    (33, "nidorino", "尼多力诺", "Poison"),
    (34, "nidoking", "尼多王", "Poison/Ground"),
    (35, "clefairy", "皮皮", "Fairy"),
    (36, "clefable", "皮可西", "Fairy"),
    (37, "vulpix", "六尾", "Fire"),
    (38, "ninetales", "九尾", "Fire"),
    (39, "jigglypuff", "胖丁", "Normal/Fairy"),
    (40, "wigglytuff", "胖可丁", "Normal/Fairy"),
    (41, "zubat", "超音蝠", "Poison/Flying"),
    (42, "golbat", "大嘴蝠", "Poison/Flying"),
    (43, "oddish", "走路草", "Grass/Poison"),
    (44, "gloom", "臭臭花", "Grass/Poison"),
    (45, "vileplume", "霸王花", "Grass/Poison"),
    (46, "paras", "派拉斯", "Bug/Grass"),
    (47, "parasect", "派拉斯特", "Bug/Grass"),
    (48, "venonat", "毛球", "Bug/Poison"),
    (49, "venomoth", "摩鲁蛾", "Bug/Poison"),
    (50, "diglett", "地鼠", "Ground"),
    (51, "dugtrio", "三地鼠", "Ground"),
    (52, "meowth", "喵喵", "Normal"),
    (53, "persian", "猫老大", "Normal"),
    (54, "psyduck", "可达鸭", "Water"),
    (55, "golduck", "哥达鸭", "Water"),
    (56, "mankey", "猴怪", "Fighting"),
    (57, "primeape", "火暴猴", "Fighting"),
    (58, "growlithe", "卡蒂狗", "Fire"),
    (59, "arcanine", "风速狗", "Fire"),
    (60, "poliwag", "蚊香蝌蚪", "Water"),
    (61, "poliwhirl", "蚊香君", "Water"),
    (62, "poliwrath", "蚊香泳士", "Water/Fighting"),
    (63, "abra", "凯西", "Psychic"),
    (64, "kadabra", "勇基拉", "Psychic"),
    (65, "alakazam", "胡地", "Psychic"),
    (66, "machop", "腕力", "Fighting"),
    (67, "machoke", "豪力", "Fighting"),
    (68, "machamp", "怪力", "Fighting"),
    (69, "bellsprout", "喇叭芽", "Grass/Poison"),
    (70, "weepinbell", "口呆花", "Grass/Poison"),
    (71, "victreebel", "大食花", "Grass/Poison"),
    (72, "tentacool", "玛瑙水母", "Water/Poison"),
    (73, "tentacruel", "毒刺水母", "Water/Poison"),
    (74, "geodude", "小拳石", "Rock/Ground"),
    (75, "graveler", "隆隆石", "Rock/Ground"),
    (76, "golem", "隆隆岩", "Rock/Ground"),
    (77, "ponyta", "小火马", "Fire"),
    (78, "rapidash", "烈焰马", "Fire"),
    (79, "slowpoke", "呆呆兽", "Water/Psychic"),
    (80, "slowbro", "呆壳兽", "Water/Psychic"),
    (81, "magnemite", "小磁怪", "Electric/Steel"),
    (82, "magneton", "三合一磁怪", "Electric/Steel"),
    (83, "farfetchd", "大葱鸭", "Normal/Flying"),
    (84, "doduo", "嘟嘟", "Normal/Flying"),
    (85, "dodrio", "嘟嘟利", "Normal/Flying"),
    (86, "seel", "小海狮", "Water"),
    (87, "dewgong", "白海狮", "Water/Ice"),
    (88, "grimer", "臭泥", "Poison"),
    (89, "muk", "臭臭泥", "Poison"),
    (90, "shellder", "大舌贝", "Water"),
    (91, "cloyster", "刺甲贝", "Water/Ice"),
    (92, "gastly", "鬼斯", "Ghost/Poison"),
    (93, "haunter", "鬼斯通", "Ghost/Poison"),
    (94, "gengar", "耿鬼", "Ghost/Poison"),
    (95, "onix", "大岩蛇", "Rock/Ground"),
    (133, "eevee", "伊布", "Normal"),
    (143, "snorlax", "卡比兽", "Normal"),
    (149, "dragonite", "快龙", "Dragon/Flying"),
    (150, "mewtwo", "超梦", "Psychic"),
    (151, "mew", "梦幻", "Psychic"),
]


def download_anime_100():
    print("\n" + "=" * 80)
    print(" 👾 [1. 动漫与角色 IP 样张下载] 正在同步 100 款经典动漫角色官方全彩立绘...")
    print("=" * 80)

    success = 0
    base_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork"
    for idx, (pid, name_en, name_zh, ptype) in enumerate(POKEMON_100, 1):
        filename = f"{idx:03d}_{name_en}.png"
        dst = ANIME_DIR / filename
        url = f"{base_url}/{pid}.png"
        if safe_download(url, dst, min_bytes=500):
            success += 1
            if success <= 5 or success % 20 == 0 or success == 100:
                print(f"  ✓ [{success}/100] 动漫角色入库: {filename} ({name_zh}) - {dst.stat().st_size} bytes")

    print(f"  ✨ 动漫角色 IP 100 款下载完毕！现地共 {success} 张高清透明立绘。")


# =============================================================================
# 2. 奢侈品品牌与 IP 品牌 (动态根据本地索引解析 100% 存在路径)
# =============================================================================
def load_svg_index():
    cache_path = ASSETS_DIR / "_temp_svgs.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def download_luxury_100(index_data):
    print("\n" + "=" * 80)
    print(" 💎 [2. 奢侈品品牌样张下载] 正在拉取 100 款国际顶级高奢官方矢量 Logo...")
    print("=" * 80)

    luxury_defs = [
        ('gucci', 'Gucci', '古驰'),
        ('chanel', 'Chanel', '香奈儿'),
        ('louis-vuitton', 'Louis Vuitton', '路易威登'),
        ('prada', 'Prada', '普拉达'),
        ('hermes', 'Hermès', '爱马仕'),
        ('dior', 'Christian Dior', '迪奥'),
        ('burberry', 'Burberry', '博柏利'),
        ('balenciaga', 'Balenciaga', '巴黎世家'),
        ('versace', 'Versace', '范思哲'),
        ('fendi', 'Fendi', '芬迪'),
        ('givenchy', 'Givenchy', '纪梵希'),
        ('celine', 'Celine', '思琳'),
        ('valentino', 'Valentino', '华伦天奴'),
        ('bottega-veneta', 'Bottega Veneta', '葆蝶家'),
        ('yves-saint-laurent', 'Yves Saint Laurent', '圣罗兰'),
        ('balmain', 'Balmain', '巴尔曼'),
        ('alexander-mcqueen', 'Alexander McQueen', '亚历山大麦昆'),
        ('loewe', 'Loewe', '罗意威'),
        ('jacquemus', 'Jacquemus', '雅克穆斯'),
        ('chloe', 'Chloé', '蔻依'),
        ('kenzo', 'Kenzo', '高田贤三'),
        ('lanvin', 'Lanvin', '浪凡'),
        ('miu-miu', 'Miu Miu', '缪缪'),
        ('moschino', 'Moschino', '莫斯奇诺'),
        ('salvatore-ferragamo', 'Salvatore Ferragamo', '菲拉格慕'),
        ('dolce-gabbana', 'Dolce & Gabbana', '杜嘉班纳'),
        ('armani', 'Giorgio Armani', '阿玛尼'),
        ('ralph-lauren', 'Ralph Lauren', '拉夫劳伦'),
        ('ermenegildo-zegna', 'Ermenegildo Zegna', '杰尼亚'),
        ('brioni', 'Brioni', '布里奥尼'),
        ('berluti', 'Berluti', '伯尔鲁帝'),
        ('moncler', 'Moncler', '盟可睐'),
        ('canada-goose', 'Canada Goose', '加拿大鹅'),
        ('supreme', 'Supreme', '至高无上'),
        ('off-white', 'Off-White', '灰白'),
        ('comme-des-garcons', 'Comme des Garçons', '川久保玲'),
        ('issey-miyake', 'Issey Miyake', '三宅一生'),
        ('vivienne-westwood', 'Vivienne Westwood', '西太后'),
        ('acne-studios', 'Acne Studios', '艾克妮'),
        ('stone-island', 'Stone Island', '石头岛'),
        ('arc-teryx', 'Arc\'teryx', '始祖鸟'),
        ('rimowa', 'Rimowa', '日默瓦'),
        ('calvin-klein', 'Calvin Klein', '卡尔文克莱恩'),
        ('tommy-hilfiger', 'Tommy Hilfiger', '汤米希尔费格'),
        ('diesel', 'Diesel', '迪赛'),
        ('lacoste', 'Lacoste', '鳄鱼'),
        ('coach', 'Coach', '蔻驰'),
        ('michael-kors', 'Michael Kors', '迈克高仕'),
        ('rolex', 'Rolex', '劳力士'),
        ('patek-philippe', 'Patek Philippe', '百达翡丽'),
        ('audemars-piguet', 'Audemars Piguet', '爱彼'),
        ('vacheron-constantin', 'Vacheron Constantin', '江诗丹顿'),
        ('jaeger-le-coultre', 'Jaeger-LeCoultre', '积家'),
        ('omega', 'Omega', '欧米茄'),
        ('iwc', 'IWC Schaffhausen', '万国表'),
        ('breitling', 'Breitling', '百年灵'),
        ('tag-heuer', 'TAG Heuer', '泰格豪雅'),
        ('hublot', 'Hublot', '宇舶'),
        ('officine-panerai', 'Panerai', '沛纳海'),
        ('chopard', 'Chopard', '萧邦'),
        ('breguet', 'Breguet', '宝玑'),
        ('piaget', 'Piaget', '伯爵'),
        ('blancpain', 'Blancpain', '宝珀'),
        ('tudor', 'Tudor', '帝舵'),
        ('longines', 'Longines', '浪琴'),
        ('tissot', 'Tissot', '天梭'),
        ('cartier', 'Cartier', '卡地亚'),
        ('tiffany-co', 'Tiffany & Co.', '蒂芙尼'),
        ('bvlgari', 'Bvlgari', '宝格丽'),
        ('harry-winston', 'Harry Winston', '海瑞温斯顿'),
        ('swarovski', 'Swarovski', '施华洛世奇'),
        ('pandora', 'Pandora', '潘多拉'),
        ('boucheron', 'Boucheron', '宝诗龙'),
        ('ferrari', 'Ferrari', '法拉利'),
        ('porsche', 'Porsche', '保时捷'),
        ('lamborghini', 'Lamborghini', '兰博基尼'),
        ('rolls-royce', 'Rolls-Royce', '劳斯莱斯'),
        ('bentley', 'Bentley', '宾利'),
        ('aston-martin', 'Aston Martin', '阿斯顿·马丁'),
        ('bugatti', 'Bugatti', '布加迪'),
        ('maserati', 'Maserati', '玛莎拉蒂'),
        ('mclaren', 'McLaren', '迈凯伦'),
        ('maybach', 'Maybach', '迈巴赫'),
        ('pagani', 'Pagani', '帕加尼'),
        ('koenigsegg', 'Koenigsegg', '科尼赛克'),
        ('lotus', 'Lotus', '路特斯'),
        ('guerlain', 'Guerlain', '娇兰'),
        ('lancome', 'Lancôme', '兰蔻'),
        ('estee-lauder', 'Estée Lauder', '雅诗兰黛'),
        ('la-mer', 'La Mer', '海蓝之谜'),
        ('byredo', 'Byredo', '百瑞德'),
        ('creed', 'Creed', '恺芮得'),
        ('montblanc', 'Montblanc', '万宝龙'),
        ('zenith', 'Zenith', '真力时'),
        ('seiko', 'Seiko', '精工'),
        ('citizen', 'Citizen', '西铁城'),
        ('hamilton', 'Hamilton', '汉米尔顿'),
        ('alfa-romeo', 'Alfa Romeo', '阿尔法·罗密欧'),
        ('jaguar', 'Jaguar', '捷豹'),
        ('land-rover', 'Land Rover', '路虎'),
        ('ray-ban', 'Ray-Ban', '雷朋'),
        ('oakley', 'Oakley', '欧克利'),
        ('fossil', 'Fossil', '化石表'),
        ('chrysler', 'Chrysler', '克莱斯勒'),
        ('mizuno', 'Mizuno', '美津浓')
    ]

    base_raw = "https://raw.githubusercontent.com/detain/svg-logos/master/svg"
    success = 0
    resolved_lux = []
    seen = set()

    for kw, name_en, name_zh in luxury_defs:
        if len(resolved_lux) >= 100:
            break
        # 精确检索 svgs.json
        key = None
        if kw in index_data and kw not in seen:
            key = kw
        else:
            cands = [x for x in index_data.keys() if (x.startswith(f'{kw}-') or x.endswith(f'-{kw}') or f'-{kw}-' in x or kw == x) and x not in seen]
            if cands:
                cands.sort(key=len)
                key = cands[0]

        if key:
            seen.add(key)
            first_c = key[0].lower()
            rel_path = f"{first_c}/{key}.svg"
            slug = kw.replace("-", "_")
            idx = len(resolved_lux) + 1
            filename = f"{idx:03d}_{slug}.svg"
            dst = LUXURY_DIR / filename

            url = f"{base_raw}/{rel_path}"
            downloaded = safe_download(url, dst, min_bytes=100)
            if not downloaded:
                simple_match = SIMPLE_ICONS_DIR / f"{kw.replace('-', '')}.svg"
                if simple_match.exists():
                    with open(simple_match, "rb") as sf, open(dst, "wb") as df:
                        df.write(sf.read())
                    downloaded = True

            if downloaded:
                success += 1
                resolved_lux.append((idx, slug, name_en, name_zh, filename))
                if success <= 5 or success % 20 == 0 or success == 100:
                    print(f"  ✓ [{success}/100] 高奢品牌入库: {filename} ({name_zh}) - {dst.stat().st_size} bytes")

    print(f"  ✨ 奢侈品品牌 100 款下载完毕！已入库 {success} 款官方矢量 Logo。")
    return resolved_lux


def download_ip_brands_100(index_data):
    print("\n" + "=" * 80)
    print(" 🎮 [3. IP 品牌样张下载] 正在拉取 100 款世界级游戏/影视/玩具/动漫 IP 官方矢量 Logo...")
    print("=" * 80)

    ip_defs = [
        ('nintendo', 'Nintendo', '任天堂'),
        ('playstation', 'PlayStation', '索尼互娱'),
        ('xbox', 'Xbox', '微软游戏'),
        ('sega', 'Sega', '世嘉'),
        ('atari', 'Atari', '雅达利'),
        ('capcom', 'Capcom', '卡普空'),
        ('konami', 'Konami', '科乐美'),
        ('bandai-namco', 'Bandai Namco', '万代南梦宫'),
        ('square-enix', 'Square Enix', '史克威尔艾尼克斯'),
        ('ubisoft', 'Ubisoft', '育碧'),
        ('electronic-arts', 'Electronic Arts (EA)', '艺电'),
        ('blizzard', 'Blizzard Entertainment', '暴雪娱乐'),
        ('riot-games', 'Riot Games', '拳头游戏'),
        ('epic-games', 'Epic Games', 'Epic游戏'),
        ('valve', 'Valve', '维尔福'),
        ('steam', 'Steam', '蒸汽平台'),
        ('rockstar', 'Rockstar Games', 'R星'),
        ('bethesda', 'Bethesda', '贝塞斯达'),
        ('cd-projekt', 'CD Projekt Red', 'CDPR'),
        ('mojang', 'Mojang Studios', '魔将工作室'),
        ('super-mario', 'Super Mario', '超级马力欧'),
        ('zelda', 'The Legend of Zelda', '塞尔达传说'),
        ('pokemon', 'Pokémon', '宝可梦'),
        ('sonic', 'Sonic the Hedgehog', '刺猬索尼克'),
        ('pac-man', 'Pac-Man', '吃豆人'),
        ('street-fighter', 'Street Fighter', '街头霸王'),
        ('resident-evil', 'Resident Evil', '生化危机'),
        ('monster-hunter', 'Monster Hunter', '怪物猎人'),
        ('tekken', 'Tekken', '铁拳'),
        ('final-fantasy', 'Final Fantasy', '最终幻想'),
        ('dragon-quest', 'Dragon Quest', '勇者斗恶龙'),
        ('metal-gear', 'Metal Gear Solid', '合金装备'),
        ('assassins-creed', 'Assassin\'s Creed', '刺客信条'),
        ('grand-theft-auto', 'Grand Theft Auto (GTA)', '侠盗猎车手'),
        ('minecraft', 'Minecraft', '我的世界'),
        ('roblox', 'Roblox', '罗布乐思'),
        ('fortnite', 'Fortnite', '堡垒之夜'),
        ('league-of-legends', 'League of Legends', '英雄联盟'),
        ('overwatch', 'Overwatch', '守望先锋'),
        ('world-of-warcraft', 'World of Warcraft', '魔兽世界'),
        ('diablo', 'Diablo', '暗黑破坏神'),
        ('fallout', 'Fallout', '辐射'),
        ('elder-scrolls', 'The Elder Scrolls', '上古卷轴'),
        ('halo', 'Halo', '光环'),
        ('cyberpunk', 'Cyberpunk 2077', '赛博朋克2077'),
        ('witcher', 'The Witcher', '巫师'),
        ('disney', 'Walt Disney', '华特迪士尼'),
        ('pixar', 'Pixar', '皮克斯'),
        ('marvel', 'Marvel', '漫威'),
        ('dc-comics', 'DC Comics', 'DC漫画'),
        ('star-wars', 'Star Wars', '星球大战'),
        ('lucasfilm', 'Lucasfilm', '卢卡斯影业'),
        ('warner-bros', 'Warner Bros.', '华纳兄弟'),
        ('universal-pictures', 'Universal Pictures', '环球影业'),
        ('paramount', 'Paramount Pictures', '派拉蒙'),
        ('dreamworks', 'DreamWorks', '梦工厂'),
        ('century-studios', '20th Century Studios', '二十世纪影业'),
        ('metro-goldwyn-mayer', 'Metro-Goldwyn-Mayer', '米高梅'),
        ('columbia-pictures', 'Columbia Pictures', '哥伦比亚影业'),
        ('studio-ghibli', 'Studio Ghibli', '吉卜力工作室'),
        ('toei-animation', 'Toei Animation', '东映动画'),
        ('batman', 'Batman', '蝙蝠侠'),
        ('superman', 'Superman', '超人'),
        ('avengers', 'Avengers', '复仇者联盟'),
        ('spider-man', 'Spider-Man', '蜘蛛侠'),
        ('iron-man', 'Iron Man', '钢铁侠'),
        ('captain-america', 'Captain America', '美国队长'),
        ('transformers', 'Transformers', '变形金刚'),
        ('autobots', 'Autobots', '汽车人'),
        ('decepticons', 'Decepticons', '霸天虎'),
        ('jurassic-park', 'Jurassic Park', '侏罗纪公园'),
        ('godzilla', 'Godzilla', '哥斯拉'),
        ('ghostbusters', 'Ghostbusters', '抓鬼敢死队'),
        ('tmnt', 'TMNT', '忍者神龟'),
        ('dragon-ball', 'Dragon Ball Z', '龙珠Z'),
        ('one-piece', 'One Piece', '海贼王'),
        ('naruto', 'Naruto', '火影忍者'),
        ('bleach', 'Bleach', '死神'),
        ('attack-on-titan', 'Attack on Titan', '进击的巨人'),
        ('demon-slayer', 'Demon Slayer', '鬼灭之刃'),
        ('sailor-moon', 'Sailor Moon', '美少女战士'),
        ('evangelion', 'Neon Genesis Evangelion', '新世纪福音战士'),
        ('gundam', 'Mobile Suit Gundam', '机动战士高达'),
        ('doraemon', 'Doraemon', '哆啦A梦'),
        ('astro-boy', 'Astro Boy', '铁臂阿童木'),
        ('yu-gi-oh', 'Yu-Gi-Oh!', '游戏王'),
        ('digimon', 'Digimon', '数码宝贝'),
        ('lego', 'Lego', '乐高'),
        ('hasbro', 'Hasbro', '孩之宝'),
        ('mattel', 'Mattel', '美泰'),
        ('barbie', 'Barbie', '芭比'),
        ('hot-wheels', 'Hot Wheels', '风火轮'),
        ('nerf', 'Nerf', '热火'),
        ('playmobil', 'Playmobil', '摩比世界'),
        ('sanrio', 'Sanrio', '三丽鸥'),
        ('hello-kitty', 'Hello Kitty', '凯蒂猫'),
        ('funko', 'Funko Pop', '方头仔'),
        ('crunchyroll', 'Crunchyroll', '脆莓动漫'),
        ('discord', 'Discord', 'Discord游戏社区'),
        ('twitch', 'Twitch', 'Twitch直播'),
        ('unity', 'Unity Engine', 'Unity游戏引擎'),
        ('unreal-engine', 'Unreal Engine', '虚幻引擎'),
        ('donkey-kong', 'Donkey Kong', '大金刚'),
        ('mortal-kombat', 'Mortal Kombat', '真人快打'),
        ('king-of-fighters', 'The King of Fighters', '拳皇'),
        ('snk', 'SNK', 'SNK电玩'),
        ('tamagotchi', 'Tamagotchi', '拓麻歌子'),
        ('beyblade', 'Beyblade', '战斗陀螺'),
        ('half-life', 'Half-Life', '半条命'),
        ('portal', 'Portal', '传送门'),
        ('team-fortress', 'Team Fortress', '军团要塞'),
        ('game-boy', 'Game Boy', '掌机'),
        ('nintendo-switch', 'Nintendo Switch', 'Switch主机'),
        ('playstation-2', 'PlayStation 2', 'PS2主机'),
        ('xbox-360', 'Xbox 360', 'Xbox360主机'),
        ('steam-deck', 'Steam Deck', 'Steam掌机'),
        ('star-trek', 'Star Trek', '星际迷航'),
        ('doctor-who', 'Doctor Who', '神秘博士'),
        ('the-simpsons', 'The Simpsons', '辛普森一家'),
        ('family-guy', 'Family Guy', '恶搞之家'),
        ('south-park', 'South Park', '南方公园'),
        ('nickelodeon', 'Nickelodeon', '尼克儿童频道'),
        ('cartoon-network', 'Cartoon Network', '卡通频道'),
        ('netflix', 'Netflix', '奈飞'),
        ('hulu', 'Hulu', '葫芦视频')
    ]

    base_raw = "https://raw.githubusercontent.com/detain/svg-logos/master/svg"
    success = 0
    resolved_ip = []
    seen = set()

    for kw, name_en, name_zh in ip_defs:
        if len(resolved_ip) >= 100:
            break

        key = None
        if kw in index_data and kw not in seen:
            key = kw
        else:
            cands = [x for x in index_data.keys() if (x.startswith(f'{kw}-') or x.endswith(f'-{kw}') or f'-{kw}-' in x or kw == x) and x not in seen]
            if cands:
                cands.sort(key=len)
                key = cands[0]

        if key:
            seen.add(key)
            first_c = key[0].lower()
            rel_path = f"{first_c}/{key}.svg"
            slug = kw.replace("-", "_")
            idx = len(resolved_ip) + 1
            filename = f"{idx:03d}_{slug}.svg"
            dst = IP_BRANDS_DIR / filename

            url = f"{base_raw}/{rel_path}"
            downloaded = safe_download(url, dst, min_bytes=100)
            if not downloaded:
                simple_match = SIMPLE_ICONS_DIR / f"{kw.replace('-', '')}.svg"
                if simple_match.exists():
                    with open(simple_match, "rb") as sf, open(dst, "wb") as df:
                        df.write(sf.read())
                    downloaded = True

            if downloaded:
                success += 1
                resolved_ip.append((idx, slug, name_en, name_zh, filename))
                if success <= 5 or success % 20 == 0 or success == 100:
                    print(f"  ✓ [{success}/100] IP 品牌入库: {filename} ({name_zh}) - {dst.stat().st_size} bytes")

    print(f"  ✨ IP 品牌 100 款下载完毕！已入库 {success} 款官方矢量 Logo。")
    return resolved_ip


# =============================================================================
# 3. 校准资产库结构与 manifests 结构化清单
# =============================================================================
def calibrate_manifests(resolved_lux, resolved_ip):
    print("\n" + "=" * 80)
    print(" 📋 [4. 资产清单与台账校准] 正在生成标准结构化 JSON 清单...")
    print("=" * 80)

    # 4.1 校准 character_ips.json
    anime_items = []
    for idx, (pid, name_en, name_zh, ptype) in enumerate(POKEMON_100, 1):
        anime_items.append({
            "id": f"anime_{idx:03d}_{name_en}",
            "dex_id": pid,
            "name_en": name_en,
            "name_zh": name_zh,
            "category": "pokemon_gen1",
            "type": ptype,
            "local_path": f"assets/character_ips/anime_100/{idx:03d}_{name_en}.png",
            "format": "transparent_png",
            "resolution": "475x475"
        })

    char_manifest = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "动漫角色 IP 资产清单 (Anime Character IPs Manifest)",
        "version": "2.0.0",
        "total_curated": len(anime_items),
        "primary_source": "PokéAPI Official Artwork (Nintendo / Creatures / GAME FREAK)",
        "items": anime_items
    }
    with open(MANIFESTS_DIR / "character_ips.json", "w", encoding="utf-8") as f:
        json.dump(char_manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 已生成校准清单: manifests/character_ips.json ({len(anime_items)} 款动漫角色)")

    # 4.2 校准 luxury_brands.json
    lux_items = []
    for idx, slug, name_en, name_zh, filename in resolved_lux:
        lux_items.append({
            "id": f"lux_{idx:03d}_{slug}",
            "name_en": name_en,
            "name_zh": name_zh,
            "slug": slug,
            "category": "luxury_fashion_horology_jewelry",
            "local_path": f"assets/brand_logos/luxury_100/{filename}",
            "format": "svg"
        })

    lux_manifest = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "国际顶级高奢品牌资产清单 (Luxury Brands Manifest)",
        "version": "1.0.0",
        "total_curated": len(lux_items),
        "sectors": ["Haute Couture", "Horology", "High Jewelry", "Luxury Automotive", "Fragrance"],
        "items": lux_items
    }
    with open(MANIFESTS_DIR / "luxury_brands.json", "w", encoding="utf-8") as f:
        json.dump(lux_manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 已生成校准清单: manifests/luxury_brands.json ({len(lux_items)} 款奢侈品)")

    # 4.3 校准 ip_brands.json
    ip_items = []
    for idx, slug, name_en, name_zh, filename in resolved_ip:
        ip_items.append({
            "id": f"ip_{idx:03d}_{slug}",
            "name_en": name_en,
            "name_zh": name_zh,
            "slug": slug,
            "category": "gaming_entertainment_toy_anime",
            "local_path": f"assets/brand_logos/ip_brands_100/{filename}",
            "format": "svg"
        })

    ip_manifest = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "全球经典 IP 品牌资产清单 (Iconic IP Brands Manifest)",
        "version": "1.0.0",
        "total_curated": len(ip_items),
        "sectors": ["Gaming Giants", "Gaming Franchises", "Entertainment Studios", "Superheroes", "Anime Franchises", "Toy & Pop"],
        "items": ip_items
    }
    with open(MANIFESTS_DIR / "ip_brands.json", "w", encoding="utf-8") as f:
        json.dump(ip_manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 已生成校准清单: manifests/ip_brands.json ({len(ip_items)} 款 IP 品牌)")

    # 清理临时缓存文件
    cache_path = ASSETS_DIR / "_temp_svgs.json"
    if cache_path.exists():
        cache_path.unlink()
        print("  ✓ 已清理临时索引缓存文件。")


def run():
    index_data = load_svg_index()
    download_anime_100()
    resolved_lux = download_luxury_100(index_data)
    resolved_ip = download_ip_brands_100(index_data)
    calibrate_manifests(resolved_lux, resolved_ip)


if __name__ == "__main__":
    run()
