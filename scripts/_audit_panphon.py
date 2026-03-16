#!/usr/bin/env python
"""Audit PanPhon feature vectors for PT-BR phoneme vocabulary."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

# Must import phonetic_features first to trigger Windows patch
import phonetic_features  # noqa: F401, E402
import panphon.featuretable  # noqa: E402

ft = panphon.featuretable.FeatureTable()
FEAT = ['syl','son','cons','cont','delrel','lat','nas','strid','voi','sg','cg',
        'ant','cor','distr','lab','hi','lo','back','round','velaric','tense',
        'long','hitone','hireg']

def vec(ph):
    segs = ft.word_fts(ph)
    return list(segs[0].numeric()) if segs else None

def cmp(a, b, label):
    va, vb = vec(a), vec(b)
    if va is None or vb is None:
        print(f"  {label}: MISSING (PanPhon does not know '{a if va is None else b}')")
        return
    diffs = [(FEAT[i], va[i], vb[i]) for i in range(24) if va[i] != vb[i]]
    ham = len(diffs)
    dist_hamming = ham / 24
    print(f"  {label}: hamming={ham}/24 ({dist_hamming:.3f}) | diff features={[d[0] for d in diffs]}")
    if va == vb:
        print("    WARNING: IDENTICAL VECTORS — PanPhon cannot distinguish these phonemes!")

print("=" * 70)
print("PanPhon Audit — PT-BR Phoneme Feature Vectors")
print("=" * 70)

print("\n--- Coverage check: all PT-BR phonemes ---")
all_phonemes = [
    'a','e','i','o','u','b','d','f','k','l','m','n','p','s','t','v','w','x','y','z',
    '\u0254',  # ɔ
    '\u0259',  # ə
    '\u025B',  # ɛ
    '\u0261',  # ɡ (IPA g, U+0261)
    '\u0263',  # ɣ
    '\u026A',  # ɪ
    '\u0272',  # ɲ
    '\u027E',  # ɾ
    '\u0283',  # ʃ
    '\u028A',  # ʊ
    '\u028E',  # ʎ
    '\u0292',  # ʒ
    '\u00E3',  # ã
    '\u00F5',  # õ
    '\u0129',  # ĩ
    '\u0169',  # ũ
    '\u1EBD',  # ẽ
    '\u1EF9',  # ỹ
    '\u028A\u0303',  # ʊ̃ (decomposed)
]
for ph in all_phonemes:
    v = vec(ph)
    status = "OK" if v and any(x != 0 for x in v) else "ZERO/MISSING"
    print(f"  {ph!r:12} (U+{ord(ph[0]):04X}) : {status}")

print("\n--- Minimal pairs: voicing (expect diff only 'voi') ---")
cmp('p','b','p/b')
cmp('t','d','t/d')
cmp('k','\u0261','k/ɡ')
cmp('f','v','f/v')
cmp('s','z','s/z')
cmp('\u0283','\u0292','ʃ/ʒ')
cmp('\u0263','x','ɣ/x  (r-coda allophone pair)')

print("\n--- Vowel height (expect diff in hi/lo/tense) ---")
cmp('i','\u026A','i/ɪ  (close vs near-close)')
cmp('u','\u028A','u/ʊ  (close vs near-close)')
cmp('e','\u025B','e/ɛ  (mid vs open-mid)')
cmp('o','\u0254','o/ɔ  (mid vs open-mid)')
cmp('a','\u0259','a/ə  (low vs mid-central/schwa)')

print("\n--- Palatal vs alveolar sonorants ---")
cmp('n','\u0272','n/ɲ')
cmp('l','\u028E','l/ʎ')
cmp('l','\u027E','l/ɾ  (lateral vs tap)')
cmp('s','\u0283','s/ʃ  (alveolar vs palatal sibilant)')
cmp('z','\u0292','z/ʒ')

print("\n--- Nasality (expect diff only 'nas') ---")
cmp('a','\u00E3','a/ã')
cmp('e','\u1EBD','e/ẽ')
cmp('i','\u0129','i/ĩ')
cmp('o','\u00F5','o/õ')
cmp('u','\u0169','u/ũ')
cmp('\u028A','\u028A\u0303','ʊ/ʊ̃')

print("\n--- Cross-class checks (should be LARGE distances) ---")
cmp('a','p','a/p  (vowel vs consonant)')
cmp('a','s','a/s  (vowel vs fricative)')
cmp('i','\u0261','i/ɡ  (vowel vs velar stop)')

print("\n--- Potential ambiguity check: x (sibilant) vs x (velar fricative) ---")
# 'x' in our corpus is the VELAR FRICATIVE (r-coda), not the IPA sibilant
# PanPhon treats ASCII 'x' as... what?
vx = vec('x')
print(f"  x (U+0078): vec={vx}")
print("  Expected: velar fricative (like ɣ but voiceless)")
print("  In PT-BR tsv: x = velar fricative (r-coda before voiceless)")
vchi = vec('\u03C7')  # χ (Greek chi) = IPA voiceless uvular fricative
if vchi:
    print(f"  χ (U+03C7, uvular): {vchi}")

print("\nDone.")
