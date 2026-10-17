"""亚文化代表样张采样下载器 (Subculture Sample Downloader)。

四个子库，统一落盘到 assets/subculture_elements/：
1. emoji_3d/        Microsoft Fluent Emoji 3D 拟物 PNG (MIT)
2. emoji_flat/      Twemoji 扁平矢量 SVG (CC-BY 4.0)
3. street_symbols/  game-icons.net 街头/暗黑/神秘学单色矢量 SVG (CC BY 3.0)，已去黑底、改黑色前景
4. memes/           经典互联网模因 (来源 ZachSaucier/fave-slackmojis，版权归原作者，仅评测研究用)

下载后做 MD5 全局去重（与 assets/ 下已有文件对比），并写出 manifests/subculture_elements.json。
用法: python3 tools/download_subculture_samples.py
"""
import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS = BASE_DIR / "assets"
OUT = ASSETS / "subculture_elements"
MANIFEST = ASSETS / "manifests" / "subculture_elements.json"
UA = {"User-Agent": "Mozilla/5.0 (Datawheel asset sampler)"}

FLUENT_RAW = "https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/"
TWEMOJI_RAW = "https://raw.githubusercontent.com/jdecked/twemoji/main/assets/svg/"
GAMEICONS_RAW = "https://raw.githubusercontent.com/game-icons/icons/master/"
SLACK_RAW = "https://raw.githubusercontent.com/ZachSaucier/fave-slackmojis/master/emojis/"

# (Fluent 目录名, 中文名, 亚文化标签)
FLUENT_3D = [
    ("Face with tears of joy", "笑哭", "reaction"),
    ("Loudly crying face", "大哭", "reaction"),
    ("Skull", "骷髅", "dark_humor"),
    ("Skull and crossbones", "骷髅交叉骨", "dark_humor"),
    ("Clown face", "小丑", "dark_humor"),
    ("Smiling face with horns", "恶魔笑", "dark_humor"),
    ("Melting face", "融化脸", "reaction"),
    ("Fire", "火焰", "hype"),
    ("Heart on fire", "燃烧的心", "hype"),
    ("Hundred points", "100分", "hype"),
    ("Sparkles", "闪光", "y2k"),
    ("Star-struck", "星星眼", "reaction"),
    ("Smiling face with sunglasses", "墨镜酷脸", "reaction"),
    ("Exploding head", "脑袋爆炸", "reaction"),
    ("Pleading face", "求求脸", "reaction"),
    ("Zany face", "滑稽脸", "reaction"),
    ("Upside-down face", "倒脸", "reaction"),
    ("Thinking face", "思考", "reaction"),
    ("Shushing face", "嘘", "reaction"),
    ("Nerd face", "书呆子", "reaction"),
    ("Face with spiral eyes", "晕眩", "reaction"),
    ("Dotted line face", "虚线脸", "reaction"),
    ("Alien", "外星人", "y2k"),
    ("Alien monster", "像素外星怪", "retro_gaming"),
    ("Ghost", "幽灵", "dark_humor"),
    ("Pile of poo", "便便", "dark_humor"),
    ("Moai", "摩艾石像", "meme_symbol"),
    ("Robot", "机器人", "cyberpunk"),
    ("Rocket", "火箭", "hype"),
    ("Money bag", "钱袋", "hype"),
    ("Gem stone", "钻石", "hype"),
    ("Crown", "皇冠", "hype"),
    ("Black heart", "黑心", "gothic"),
    ("Broken heart", "心碎", "gothic"),
    ("Peace symbol", "和平符号", "occult_spiritual"),
    ("Yin yang", "阴阳", "occult_spiritual"),
    ("Crystal ball", "水晶球", "occult_spiritual"),
    ("Nazar amulet", "恶魔之眼护符", "occult_spiritual"),
    ("Eye", "眼睛", "occult_spiritual"),
    ("Joker", "鬼牌小丑", "gothic"),
    ("Cherries", "樱桃", "y2k"),
    ("Sign of the horns", "金属礼手势", "rock_punk"),
    ("Middle finger", "中指", "rock_punk"),
]

# (codepoint, 中文名, 标签) —— 扁平矢量版本，用于与 3D 版做“同语义跨画风”对照
TWEMOJI_FLAT = [
    ("1f602", "笑哭(扁平)", "reaction"),
    ("1f480", "骷髅(扁平)", "dark_humor"),
    ("1f921", "小丑(扁平)", "dark_humor"),
    ("1f525", "火焰(扁平)", "hype"),
    ("2728", "闪光(扁平)", "y2k"),
    ("1f5a4", "黑心(扁平)", "gothic"),
    ("262e", "和平符号(扁平)", "occult_spiritual"),
    ("262f", "阴阳(扁平)", "occult_spiritual"),
    ("26a1", "闪电(扁平)", "rock_punk"),
    ("1f47d", "外星人(扁平)", "y2k"),
    ("1f344", "蘑菇(扁平)", "psychedelic"),
    ("1f308", "彩虹(扁平)", "psychedelic"),
]

# (仓库相对路径, 中文名, 标签)
GAME_ICONS = [
    ("lorc/skull-crossed-bones.svg", "骷髅交叉骨", "rock_punk"),
    ("sbed/death-skull.svg", "死亡骷髅", "gothic"),
    ("skoll/burning-skull.svg", "燃烧骷髅", "rock_punk"),
    ("lorc/crowned-skull.svg", "王冠骷髅", "gothic"),
    ("lorc/horned-skull.svg", "羊角骷髅", "gothic"),
    ("lorc/daemon-skull.svg", "恶魔骷髅", "gothic"),
    ("lorc/alien-skull.svg", "外星骷髅", "y2k"),
    ("lorc/spade-skull.svg", "黑桃骷髅", "gothic"),
    ("delapouite/dagger-rose.svg", "匕首玫瑰", "tattoo_flash"),
    ("lorc/chained-heart.svg", "锁链之心", "tattoo_flash"),
    ("lorc/crowned-heart.svg", "王冠之心", "tattoo_flash"),
    ("delapouite/heart-wings.svg", "翼之心", "tattoo_flash"),
    ("lorc/broken-heart.svg", "破碎之心", "gothic"),
    ("lorc/winged-sword.svg", "翼剑", "tattoo_flash"),
    ("lorc/crossed-swords.svg", "交叉双剑", "tattoo_flash"),
    ("lorc/anchor.svg", "船锚", "tattoo_flash"),
    ("delapouite/pirate-flag.svg", "海盗旗", "rock_punk"),
    ("lorc/barbed-wire.svg", "铁丝网", "y2k_tribal"),
    ("lorc/thorny-vine.svg", "荆棘藤", "y2k_tribal"),
    ("lorc/crown-of-thorns.svg", "荆棘冠", "gothic"),
    ("lorc/spider-web.svg", "蛛网", "gothic"),
    ("lorc/bat-wing.svg", "蝙蝠翼", "gothic"),
    ("lorc/angel-wings.svg", "天使之翼", "y2k"),
    ("carl-olsen/flame.svg", "火焰图腾", "hot_rod"),
    ("delapouite/all-seeing-eye.svg", "全视之眼", "occult_esoteric"),
    ("delapouite/eye-of-horus.svg", "荷鲁斯之眼", "occult_esoteric"),
    ("lorc/third-eye.svg", "第三只眼", "occult_esoteric"),
    ("lorc/ouroboros.svg", "衔尾蛇", "occult_esoteric"),
    ("lorc/pentagram-rose.svg", "五芒星玫瑰", "occult_esoteric"),
    ("lorc/ankh.svg", "生命之符", "occult_esoteric"),
    ("delapouite/triquetra.svg", "三一结", "occult_esoteric"),
    ("delapouite/yin-yang.svg", "阴阳(线描)", "occult_esoteric"),
    ("lorc/lotus.svg", "莲花", "occult_esoteric"),
    ("delapouite/peace-dove.svg", "和平鸽", "occult_esoteric"),
    ("lorc/biohazard.svg", "生化危险", "cyberpunk"),
    ("badges/radioactive.svg", "辐射警告", "cyberpunk"),
    ("delapouite/eight-ball.svg", "黑8台球", "streetwear"),
    ("delapouite/devil-mask.svg", "恶魔面具", "gothic"),
    ("lorc/lips.svg", "红唇", "streetwear"),
]

# (文件名, 中文名, 标签) —— 仅收 PNG；JPG 无 Alpha 通道，标记 needs_matting
MEMES = [
    ("doge-laugh.png", "神烦狗大笑 (Doge)", "classic_meme"),
    ("evil-kermit.png", "邪恶科米蛙 (Evil Kermit)", "classic_meme"),
    ("kermit-tea.png", "科米蛙喝茶 (But That's None of My Business)", "classic_meme"),
    ("kermit-disbelief.png", "科米蛙震惊", "classic_meme"),
    ("thonking.png", "Thonk 思考脸", "classic_meme"),
    ("spiderman-pointing.png", "蜘蛛侠互指 (Spider-Man Pointing)", "classic_meme"),
    ("take-my-money.png", "快拿走我的钱 (Shut Up And Take My Money)", "classic_meme"),
    ("too-damn-high.png", "太他妈高了 (Too Damn High)", "classic_meme"),
    ("always-has-been.png", "一直都是 (Always Has Been)", "classic_meme"),
    ("darkside.png", "黑暗面 (Come to the Dark Side)", "classic_meme"),
    ("sad-cat.png", "悲伤猫 (Sad Cat)", "classic_meme"),
    ("sus.png", "可疑 (Among Us Sus)", "classic_meme"),
    ("homer-doh.png", "荷马 D'oh", "classic_meme"),
    ("patrick-shocked.png", "派大星震惊", "classic_meme"),
    ("lord-farquad-interested.png", "法奎德领主", "classic_meme"),
    ("keanu-woah.png", "基努 Woah", "classic_meme"),
    ("not-listening-smeagol.png", "咕噜不听", "classic_meme"),
    ("uh-oh-kirby.png", "卡比 Uh-oh", "classic_meme"),
    ("graphic-design-is-my-passion.png", "平面设计是我的热情", "classic_meme"),
    ("harold-headshot.jpg", "隐藏痛苦的哈罗德 (Hide the Pain Harold)", "classic_meme"),
    ("disaster-girl.jpg", "灾难女孩 (Disaster Girl)", "classic_meme"),
]


def fetch(url: str, retries: int = 2) -> bytes | None:
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        time.sleep(1 + i)
    return None


def png_size(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return f"{w}x{h}"
    return "vector" if data.lstrip()[:4] in (b"<svg", b"<?xm") else "unknown"


def png_has_alpha(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n" and data[25] in (4, 6) or b"tRNS" in data[:4096]


def clean_game_icon(svg: bytes) -> bytes:
    """game-icons 原图是黑底白图：去掉 512x512 黑色背景，前景改为纯黑，得到透明背景单色矢量。"""
    s = svg.decode("utf-8")
    s = re.sub(r'<path d="M0 0h512v512H0z"\s*/>', "", s)
    s = s.replace('fill="#fff"', 'fill="#000"')
    return s.encode("utf-8")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def existing_hashes() -> dict[str, str]:
    out = {}
    for p in ASSETS.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and OUT not in p.parents and p.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}:
            out[hashlib.md5(p.read_bytes()).hexdigest()] = str(p.relative_to(BASE_DIR))
    return out


def main():
    seen = existing_hashes()
    items, skipped, failed = [], [], []

    def save(sub: str, fname: str, data: bytes, meta: dict):
        h = hashlib.md5(data).hexdigest()
        if h in seen:
            skipped.append((fname, seen[h]))
            return
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / fname
        p.write_bytes(data)
        rel = str(p.relative_to(BASE_DIR))
        seen[h] = rel
        alpha = True if fname.endswith(".svg") else png_has_alpha(data)
        items.append({**meta, "local_path": rel, "resolution": png_size(data),
                      "transparent": alpha, "needs_matting": not alpha, "md5": h})

    # 1. Fluent 3D
    for idx, (name, zh, tag) in enumerate(FLUENT_3D, 1):
        s = slug(name).replace("_", "_")
        snake = name.lower().replace(" ", "_")
        cands = [f"assets/{name}/3D/{snake}_3d.png", f"assets/{name}/Default/3D/{snake}_3d_default.png"]
        data = None
        for c in cands:
            data = fetch(FLUENT_RAW + urllib.parse.quote(c))
            if data:
                break
        if not data:
            failed.append(name); continue
        save("emoji_3d", f"{idx:02d}_{s}.png", data, {
            "id": f"sub_emoji3d_{s}", "group": "emoji_3d", "name_en": name, "name_zh": zh,
            "subculture_tag": tag, "source": "microsoft/fluentui-emoji", "license": "MIT"})
        print(f"[emoji_3d] {zh}")

    # 2. Twemoji flat
    for idx, (cp, zh, tag) in enumerate(TWEMOJI_FLAT, 1):
        data = fetch(TWEMOJI_RAW + f"{cp}.svg")
        if not data:
            failed.append(cp); continue
        save("emoji_flat", f"{idx:02d}_u{cp}.svg", data, {
            "id": f"sub_emojiflat_{cp}", "group": "emoji_flat", "name_en": f"U+{cp.upper()}", "name_zh": zh,
            "subculture_tag": tag, "source": "jdecked/twemoji", "license": "CC-BY-4.0"})
        print(f"[emoji_flat] {zh}")

    # 3. game-icons street symbols
    for idx, (path, zh, tag) in enumerate(GAME_ICONS, 1):
        data = fetch(GAMEICONS_RAW + path)
        if not data:
            failed.append(path); continue
        data = clean_game_icon(data)
        stem = Path(path).stem
        save("street_symbols", f"{idx:02d}_{stem.replace('-', '_')}.svg", data, {
            "id": f"sub_symbol_{stem.replace('-', '_')}", "group": "street_symbols", "name_en": stem,
            "name_zh": zh, "subculture_tag": tag, "source": f"game-icons.net ({path.split('/')[0]})",
            "license": "CC-BY-3.0"})
        print(f"[street_symbols] {zh}")

    # 4. memes
    for idx, (fname, zh, tag) in enumerate(MEMES, 1):
        data = fetch(SLACK_RAW + fname)
        if not data:
            failed.append(fname); continue
        ext = Path(fname).suffix.lower()
        stem = Path(fname).stem.replace("-", "_")
        save("memes", f"{idx:02d}_{stem}{ext}", data, {
            "id": f"sub_meme_{stem}", "group": "memes", "name_en": Path(fname).stem, "name_zh": zh,
            "subculture_tag": tag, "source": "ZachSaucier/fave-slackmojis",
            "license": "unverified_third_party (research/eval only)"})
        print(f"[memes] {zh}")

    groups = {}
    for it in items:
        groups[it["group"]] = groups.get(it["group"], 0) + 1
    manifest = {
        "title": "Datawheel 亚文化代表元素样张清单 (Emoji / 模因 / 街头与神秘学符号)",
        "version": "1.0.0",
        "total_curated": len(items),
        "group_counts": groups,
        "notes": [
            "emoji_3d 与 emoji_flat 存在同语义跨画风对照对 (骷髅/小丑/火焰/阴阳/和平等)，可用于跨风格身份一致性评测",
            "street_symbols 已去除 game-icons 原生黑色背景并转为黑色前景，可直接按需重新着色",
            "memes 版权归原作者，仅限内部评测研究；needs_matting=true 的条目需抠图后才能用于印花合成",
        ],
        "items": items,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总计入库: {len(items)}  分组: {groups}")
    print(f"MD5 重复跳过: {len(skipped)} {skipped}")
    print(f"下载失败: {len(failed)} {failed}")


if __name__ == "__main__":
    main()
