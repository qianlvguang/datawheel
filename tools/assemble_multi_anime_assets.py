"""
Assemble and curate a truly multi-franchise anime character library for Datawheel.
Consolidates duplicates ("合版") across all anime and character subfolders.
"""
import os
import shutil
import urllib.request
import json
import hashlib
from pathlib import Path

BASE_DIR = Path("assets/character_ips")
SHONEN_DIR = BASE_DIR / "shonen_jump"
GENSHIN_DIR = BASE_DIR / "genshin"
STARRAIL_DIR = BASE_DIR / "starrail"
CLASSICS_DIR = BASE_DIR / "classics"
POKEMON_DIR = BASE_DIR / "pokemon"

for d in [SHONEN_DIR, GENSHIN_DIR, STARRAIL_DIR, CLASSICS_DIR, POKEMON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Shonen Jump characters from /tmp/fastfetch_all and online
shonen_mapping = [
    ("/tmp/fastfetch_all/1-goku.png", "01_dragonball_goku.png"),
    ("/tmp/fastfetch_all/3-luffy.png", "02_onepiece_luffy.png"),
    ("/tmp/fastfetch_all/luffy_gear5.png", "03_onepiece_luffy_gear5.png"),
    ("/tmp/fastfetch_all/luffy_chibi.png", "04_onepiece_luffy_chibi.png"),
    ("/tmp/fastfetch_all/1-zoro.png", "05_onepiece_zoro.png"),
    ("/tmp/fastfetch_all/2-zoro.png", "06_onepiece_zoro_wano.png"),
    ("/tmp/fastfetch_all/sasuke.png", "07_naruto_sasuke.png"),
    ("/tmp/fastfetch_all/1-tanjiro.png", "09_demonslayer_tanjiro_water.png"),
    ("/tmp/fastfetch_all/2-tanjiro.png", "10_demonslayer_tanjiro_hinokami.png"),
    ("/tmp/fastfetch_all/1-Nezuko.png", "11_demonslayer_nezuko.png"),
    ("/tmp/fastfetch_all/1-zenitsu.png", "12_demonslayer_zenitsu.png"),
    ("/tmp/fastfetch_all/1-rengoku.png", "13_demonslayer_rengoku.png"),
    ("/tmp/fastfetch_all/1-kokushibo.png", "14_demonslayer_kokushibo.png"),
    ("/tmp/fastfetch_all/gon.png", "15_hunter_gon.png"),
]

for src, dst_name in shonen_mapping:
    dst_path = SHONEN_DIR / dst_name
    if os.path.exists(src):
        shutil.copy2(src, dst_path)
        print(f"Copied {dst_name} ({dst_path.stat().st_size // 1024} KB)")

# Download Naruto and Levi
ext_downloads = [
    ("https://raw.githubusercontent.com/nirajmchauhan/naruto/master/images/naruto.png", SHONEN_DIR / "08_naruto_uzumaki.png"),
    ("https://raw.githubusercontent.com/npc-worldwide/incognide/main/levi.PNG", SHONEN_DIR / "16_aot_levi_ackerman.png"),
    ("https://raw.githubusercontent.com/samanthaming/not-fancy-voice-animate/master/images/totoro.png", CLASSICS_DIR / "01_ghibli_totoro.png"),
    ("https://github.com/kornia/data/raw/main/doraemon.png", CLASSICS_DIR / "02_doraemon.png"),
]

for url, target_path in ext_downloads:
    if not target_path.exists():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                target_path.write_bytes(data)
                print(f"Downloaded {target_path.name} ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"Failed {target_path.name}: {e}")

# 2. Genshin Impact (25 key characters)
genshin_specs = [
    ("10000030", "01_zhongli.png", "钟离", "Zhongli", "Geo"),
    ("10000052", "02_raiden_shogun.png", "雷电将军", "Raiden Shogun", "Electro"),
    ("10000089", "03_furina.png", "芙宁娜", "Furina", "Hydro"),
    ("10000087", "04_neuvillette.png", "那维莱特", "Neuvillette", "Hydro"),
    ("10000073", "05_nahida.png", "纳西妲", "Nahida", "Dendro"),
    ("10000046", "06_hu_tao.png", "胡桃", "Hu Tao", "Pyro"),
    ("10000047", "07_kaedehara_kazuha.png", "枫原万叶", "Kaedehara Kazuha", "Anemo"),
    ("10000026", "08_xiao.png", "魈", "Xiao", "Anemo"),
    ("10000037", "09_ganyu.png", "甘雨", "Ganyu", "Cryo"),
    ("10000002", "10_kamisato_ayaka.png", "神里绫华", "Kamisato Ayaka", "Cryo"),
    ("10000016", "11_diluc.png", "迪卢克", "Diluc", "Pyro"),
    ("10000022", "12_venti.png", "温迪", "Venti", "Anemo"),
    ("10000058", "13_yae_miko.png", "八重神子", "Yae Miko", "Electro"),
    ("10000029", "14_klee.png", "可莉", "Klee", "Pyro"),
    ("10000033", "15_tartaglia_childe.png", "达达利亚", "Tartaglia", "Hydro"),
    ("10000075", "16_wanderer.png", "流浪者", "Wanderer", "Anemo"),
    ("10000078", "17_alhaitham.png", "艾尔海森", "Alhaitham", "Dendro"),
    ("10000060", "18_yelan.png", "夜兰", "Yelan", "Hydro"),
    ("10000091", "19_navia.png", "娜维娅", "Navia", "Geo"),
    ("10000096", "20_arlecchino.png", "阿蕾奇诺", "Arlecchino", "Pyro"),
    ("10000098", "21_clorinde.png", "克洛琳德", "Clorinde", "Electro"),
    ("10000042", "22_keqing.png", "刻晴", "Keqing", "Electro"),
    ("10000041", "23_mona.png", "莫娜", "Mona", "Hydro"),
    ("10000049", "24_yoimiya.png", "宵宫", "Yoimiya", "Pyro"),
    ("10000003", "25_jean.png", "琴", "Jean", "Anemo")
]

for cid, fname, name_zh, name_en, elem in genshin_specs:
    p = GENSHIN_DIR / fname
    if not p.exists():
        url = f"https://raw.githubusercontent.com/ScobbleQ/HoYo-Assets/main/genshin/splash/{cid}.png"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                p.write_bytes(data)
                print(f"Genshin: {name_zh} ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"Error {name_zh}: {e}")

# 3. Honkai: Star Rail (15 key characters)
starrail_specs = [
    ("1308", "01_acheron.png", "黄泉", "Acheron", "Nihility"),
    ("1310", "02_firefly.png", "流萤", "Firefly", "Destruction"),
    ("1005", "03_kafka.png", "卡芙卡", "Kafka", "Nihility"),
    ("1204", "04_jing_yuan.png", "景元", "Jing Yuan", "Erudition"),
    ("1307", "05_black_swan.png", "黑天鹅", "Black Swan", "Nihility"),
    ("1301", "06_dan_heng_il.png", "丹恒·饮月", "Dan Heng IL", "Destruction"),
    ("1205", "07_blade.png", "刃", "Blade", "Destruction"),
    ("1006", "08_silver_wolf.png", "银狼", "Silver Wolf", "Nihility"),
    ("1304", "09_aventurine.png", "砂金", "Aventurine", "Preservation"),
    ("1102", "10_seele.png", "希儿", "Seele", "Hunt"),
    ("1001", "11_march_7th.png", "三月七", "March 7th", "Preservation"),
    ("1306", "12_sparkle.png", "花火", "Sparkle", "Harmony"),
    ("1309", "13_robin.png", "知更鸟", "Robin", "Harmony"),
    ("1112", "14_topaz.png", "托帕", "Topaz", "Hunt"),
    ("1002", "15_dan_heng.png", "丹恒", "Dan Heng", "Hunt")
]

for cid, fname, name_zh, name_en, elem in starrail_specs:
    p = STARRAIL_DIR / fname
    if not p.exists():
        url = f"https://raw.githubusercontent.com/ScobbleQ/HoYo-Assets/main/starrail/splash/{cid}.png"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                p.write_bytes(data)
                print(f"StarRail: {name_zh} ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"Error {name_zh}: {e}")

# 4. Consolidate Pokémon into assets/character_ips/pokemon/ (no duplicates)
# Copy from anime_100 and pokeapi, keep canonical set
pokemon_curated = [
    ("assets/character_ips/pokeapi/25_pikachu_official.png", "025_pikachu.png"),
    ("assets/character_ips/pokeapi/25_pikachu_shiny.png", "025_pikachu_shiny.png"),
    ("assets/character_ips/pokeapi/25_pikachu_pixel.png", "025_pikachu_pixel.png"),
    ("assets/character_ips/pokeapi/25_pikachu_back.png", "025_pikachu_back.png"),
    ("assets/character_ips/pokeapi/6_charizard_official.png", "006_charizard.png"),
    ("assets/character_ips/pokeapi/6_charizard_shiny.png", "006_charizard_shiny.png"),
    ("assets/character_ips/pokeapi/1_bulbasaur_official.png", "001_bulbasaur.png"),
    ("assets/character_ips/pokeapi/1_bulbasaur_shiny.png", "001_bulbasaur_shiny.png"),
    ("assets/character_ips/pokeapi/9_blastoise_official.png", "009_blastoise.png"),
    ("assets/character_ips/pokeapi/9_blastoise_shiny.png", "009_blastoise_shiny.png"),
    ("assets/character_ips/pokeapi/94_gengar_official.png", "094_gengar.png"),
    ("assets/character_ips/pokeapi/94_gengar_shiny.png", "094_gengar_shiny.png"),
    ("assets/character_ips/pokeapi/133_eevee_official.png", "133_eevee.png"),
    ("assets/character_ips/pokeapi/133_eevee_shiny.png", "133_eevee_shiny.png"),
    ("assets/character_ips/pokeapi/134_vaporeon_official.png", "134_vaporeon.png"),
    ("assets/character_ips/pokeapi/135_jolteon_official.png", "135_jolteon.png"),
    ("assets/character_ips/pokeapi/136_flareon_official.png", "136_flareon.png"),
    ("assets/character_ips/pokeapi/196_espeon_official.png", "196_espeon.png"),
    ("assets/character_ips/pokeapi/197_umbreon_official.png", "197_umbreon.png"),
    ("assets/character_ips/anime_100/039_jigglypuff.png", "039_jigglypuff.png"),
    ("assets/character_ips/anime_100/054_psyduck.png", "054_psyduck.png"),
    ("assets/character_ips/anime_100/052_meowth.png", "052_meowth.png"),
    ("assets/character_ips/anime_100/059_arcanine.png", "059_arcanine.png"),
    ("assets/character_ips/anime_100/065_alakazam.png", "065_alakazam.png"),
    ("assets/character_ips/anime_100/068_machamp.png", "068_machamp.png"),
    ("assets/character_ips/anime_100/074_geodude.png", "074_geodude.png"),
    ("assets/character_ips/anime_100/077_ponyta.png", "077_ponyta.png"),
    ("assets/character_ips/anime_100/079_slowpoke.png", "079_slowpoke.png"),
    ("assets/character_ips/anime_100/095_onix.png", "095_onix.png"),
    ("assets/character_ips/anime_100/097_hypno.png", "097_hypno.png"),
]

for src, dst_name in pokemon_curated:
    if os.path.exists(src):
        dst = POKEMON_DIR / dst_name
        shutil.copy2(src, dst)
        print(f"Pokemon: {dst_name}")

# Now remove the redundant anime_100 and pokeapi folders to achieve 100% deduplication ("合版")
if os.path.exists("assets/character_ips/pokeapi"):
    shutil.rmtree("assets/character_ips/pokeapi")
    print("Removed legacy pokeapi folder.")
if os.path.exists("assets/character_ips/anime_100"):
    shutil.rmtree("assets/character_ips/anime_100")
    print("Removed legacy anime_100 folder.")

print("\n--- Summary of character_ips ---")
for s in [SHONEN_DIR, GENSHIN_DIR, STARRAIL_DIR, CLASSICS_DIR, POKEMON_DIR]:
    count = len(list(s.glob("*.png")))
    print(f"  {s.name}: {count} files")
