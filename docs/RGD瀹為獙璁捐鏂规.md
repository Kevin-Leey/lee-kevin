# RGD 瀹為獙璁捐涓庤瘉鎹竟鐣孿n
鏇存柊鏃ユ湡锛?026-07-17

## 1. 鐮旂┒闂涓庨獙璇侀『搴廫n
鏈枃鍙楠屼竴涓牳蹇冨懡棰橈細**褰撳墠鍦烘櫙闇€瑕佹洿澶氭帹鐞嗭紝涓嶇瓑浜庢參鎺ㄧ悊杩斿洖鏃朵粛鏈夊彲鎵ц鐨勭籂姝ｆ満浼氥€?* 鍥犳锛屽疄楠屽厛楠岃瘉 allocator 閫夋嫨鐨?release state 鏄惁鍖呭惈 outcome-improving action锛屽啀鎶ュ憡 slow executor銆乺elease arbitration 鍜岄棴鐜帶鍒跺叡鍚屼綔鐢ㄥ悗鐨勭郴缁熺鐐广€俓n
璇佹嵁鎸変互涓嬮『搴忕粍缁囷細

1. fresh holdout matched-action rollout锛氱洿鎺ユ楠?post-latency corrective opportunity锛沑n2. fixed-query delay锛氬湪鍚屼竴 query cohort 涓婃楠屾満浼氶殢 release delay 鐨勬€讳綋鍙樺寲锛沑n3. seed-matched closed loop锛氭姤鍛婂畬鏁寸郴缁熺殑鏂瑰悜鎬х鐐癸紱
4. online lifecycle锛氬尯鍒?query銆乺eturn銆乽navailable銆乺ewritten 鍜?kept锛沑n5. 4/5/6 杞﹂亾銆乨ensity 2/3锛氭姤鍛婇浂闄勫姞寤惰繜涓嬬殑浜ら€氬帇鍔涜竟鐣屻€俓n
闂幆涓荤鐐逛笉鏄?powered superiority test銆俿eeds 100--129 鍦?hidden-slower bridge 淇鏈熼棿宸茬粡鍙锛屾晠鍙綔 descriptive operating point銆傜湡姝ｆ湭瑙︾鐨勪富鏈哄埗楠岃瘉浣跨敤 seeds 160--189銆俓n
## 2. 缁熶竴杩愯鍗忚

- 鐜锛歚highway-v0`锛沺olicy/simulation 鍧囦负 10 Hz锛沞pisode 30 s锛涘垵濮嬮€熷害 26 m/s锛? 鏉¤溅閬撱€?0 杈嗚溅涓?nominal 璁剧疆銆俓n- 绂绘暎鍔ㄤ綔锛歭ane left銆乲eep銆乴ane right銆乫aster銆乻lower銆俓n- slow executor锛歚Qwen/Qwen3-8B`锛泃emperature 0锛?4 output tokens锛?20 s timeout銆俓n- 鍏变韩鎺у埗锛氱浉鍚?complete fast policy銆乹uery map銆乻ix-call budget銆?0-frame cooldown銆乽nified safety arbitration 涓?actuation adapter銆俓n- 涓诲欢杩燂細1.7 s锛屽搴?17 涓?policy steps銆俻ending 鏈熼棿姣忎竴姝ラ噸鏂拌绠?complete fast policy锛屼笉鍐荤粨 query-frame 鍔ㄤ綔銆俓n- RGD锛歱riority threshold 0.16锛宱pportunity floor 0.20銆俓n- comparators锛歊andom 0.02锛孶ncertainty exposure 0.07锛孴TC-risk cutoff 0.43锛汿TC-delay threshold 0.208銆俓n- hidden-slower bridge锛氭墍鏈夋渶缁堣繍琛屽潎鍚敤銆傚畠鍙湪 closing risk 涓嬮噸鏂版毚闇茶 simulator 鐩爣閫熷害涓嬮檺闅愯棌鐨?`SLOWER`锛屽苟鏄犲皠鍒拌緝浣庣墿鐞嗙洰鏍囬€熷害锛涙墍鏈?allocator 鍏辩敤鐩稿悓 bridge銆俓n- transfer锛?/5/6 lanes 脳 `vehicles_density` 2.0/3.0锛宻eeds 0--29锛屼笉娉ㄥ叆棰濆寤惰繜銆傝鍙傛暟鍙嶅悜缂╂斁鍒濆闂磋窛锛屼笉绛夊悓浜?vehicles/km銆俓n- 缁熻鍗曚綅锛歴imulator seed銆備竴涓?seed 鍐呯殑澶氫釜 query 涓嶈兘褰撲綔鐙珛鏍锋湰銆俓n
Seed 鍒掑垎锛歕n
| 鐢ㄩ€?| Seeds | 鐘舵€?|
|---|---:|---|
| 闃堝€间笌 exposure 鏍″噯 | 30--59 | calibration |
| 鎻忚堪鎬т富闂幆 | 100--129 | visible during bridge repair |
| 闂幆 latency sweep | 130--159 | disjoint endpoint set |
| fresh mechanism holdout | 160--189 | first inspected after protocol freeze |
| lane/density stress | 0--29 | zero-added-delay transfer grid |

## 3. 涓绘満鍒跺疄楠岋細fresh release-state rollouts

杈撳叆涓?seeds 160--189 鐨?Fast-only 杞ㄨ抗銆俿low model 涓嶅弬涓庢瀹為獙锛岄伩鍏嶆妸 executor quality 娣峰叆 allocator construct銆傚浜庢瘡涓?allocator 閫変腑鐨?query锛歕n
1. 鎸夐娴?delay 鎺ㄨ繘 complete fast policy锛屽緱鍒?release state锛沑n2. 鏋氫妇璇ョ姸鎬佺殑鍏ㄩ儴 runtime-admissible actions锛沑n3. 姣忎釜 action 閮界粡杩囩浉鍚?safety map 涓?hidden-slower actuation adapter锛沑n4. 鎵ц璇?effective action 涓€娆★紝闅忓悗璺熼殢鐩稿悓 fast policy 20 steps锛沑n5. 鐢ㄦ姌鎵ｅ綊涓€鍖?simulator return 鍑?collision indicator锛岃绠楃浉瀵?matched-fast continuation 鐨?advantage锛沑n6. 鑻ュ瓨鍦ㄤ笉鍚屼簬 effective fast action 涓?advantage 鑷冲皯涓?0.02 鐨?action锛屽垯 corrective set 闈炵┖銆俓n
閿佸畾鍙傛暟涓?`H=20`銆乣gamma=.99`銆乣epsilon=.02`銆俥ffective action identity 鍚屾椂鍖呭惈 safety 鍚庣鏁ｆ寚浠や笌 hidden-slower bridge 鐨?target-speed side effect銆侳ast prefix 閲嶆斁鐨勬渶澶т綅缃宸负 0 m銆俓n
涓荤粨鏋滐細

| Allocator | Corrective / release | Fraction | 95% seed-cluster CI |
|---|---:|---:|---:|
| RGD | 11/81 | .136 | [.082, .184] |
| TTC-delay | 3/95 | .032 | [.000, .071] |
| TTC-risk | 2/80 | .025 | [.000, .063] |

RGD--TTC-delay difference 涓?.104 [.049, .155]锛汻GD--TTC-risk difference 涓?.111 [.048, .167]銆俆TC-delay 宸蹭娇鐢ㄤ笌 RGD 鐩稿悓鐨?latency-survival 淇℃伅锛屽洜姝ゅ樊寮備笉鑳戒粎褰掑洜浜庘€滅煡閬撳欢杩熲€濓紝鑰屼笌 admissible alternatives 鍜?recovery headroom 鐨勮仈鍚堢瓫閫変竴鑷淬€俓n
Margin sensitivity 涓嶉噸鏂拌繍琛?simulator銆傛妸鍚屼竴 rollout 鐨?label 鏀逛负 `epsilon=.01/.02/.05` 鍚庯紝RGD--TTC-delay differences 涓?.157/.104/.065锛屽搴?95% intervals [.079, .232]銆乕.049, .155]銆乕.024, .105]銆俓n
## 4. Fixed-query delay 涓庨棴鐜竟鐣孿n
Fixed-query 瀹為獙鍥哄畾 76 涓?RGD query states锛屽彧鎶?release delay 鏀逛负 0.7銆?.7銆?.7 s锛歕n
| Release delay | Corrective / query | Fraction |
|---:|---:|---:|
| 0.7 s | 18/76 | .237 |
| 1.7 s | 10/76 | .132 |
| 2.7 s | 8/76 | .105 |

66/76 鐨?binary pattern 绋冲畾鎴栧崟璋冪缉灏忥紱10/76 闈炲崟璋冿紝鍥犱负鍔ㄦ€佷氦閫?gap 鍙兘鍏抽棴鍚庨噸鏂版墦寮€銆傚洜姝ゆ鏂囧彧澹版槑 aggregate erosion锛屼笉澹版槑 universal monotonicity銆俓n
闂幆 latency endpoint 浣跨敤 disjoint seeds 130--159锛歕n
| Added delay | Dis. (m) | Coll. | Suc. | Slow calls |
|---:|---:|---:|---:|---:|
| 0.0 s | 597.10 | .033 | 29/30 | 131 |
| 0.7 s | 594.77 | .033 | 29/30 | 123 |
| 1.7 s | 586.44 | .033 | 29/30 | 90 |
| 2.7 s | 563.26 | .100 | 27/30 | 45 |

璇?sweep 鍙繍琛?RGD锛屼笉璇嗗埆 allocator 脳 latency interaction锛屼篃涓嶆楠?latency prediction error 鎴?jitter銆俓n
## 5. 闂幆銆乴ifecycle 涓庝氦閫氬帇鍔沑n
鎻忚堪鎬т富闂幆浣跨敤 seeds 100--129銆俁GD 涓?28/30锛孴TC-risk 涓?27/30锛屽叾浣欐帶鍒朵负 26/30銆俁GD 鐨?observed collision rate 涓?.067锛屼絾 paired intervals 瀹戒笖 Holm-adjusted exact tests 涓嶆敮鎸?completion superiority銆傝鏂囧彧鑳芥妸璇ヨ〃浣滀负涓庢満鍒舵柟鍚戜竴鑷寸殑绯荤粺绔偣銆俓n
Lifecycle 浣跨敤 query銆乺eturn銆乽navailable銆乺ewritten 鍜?kept 浜斾釜浜嬩欢銆俙kept` 鍙〃绀?release 鍚庝粛涓庡綋鍓?fast action 涓嶅悓锛屼笉琛ㄧず reward 鎴?safety improvement銆備富鏂囧繀椤绘妸 execution eligibility 涓?outcome-grounded corrective opportunity 鍒嗗紑銆俓n
4/5/6 杞﹂亾銆乨ensity 2/3 浣跨敤 540 涓?zero-added-delay episodes銆俁GD 鍦?density 2.0 涓?5/90 collisions锛屽湪 density 3.0 涓?37/90锛汿TC-risk 涓?Fast-only 鍚屾牱浠庝綆纰版挒璺冨崌鍒?35/90 鍜?34/90銆俧rame-zero median forward gap 浠?10.55/9.31/8.21 m 缂╁皬鍒?7.03/6.21/5.48 m锛屾瘮渚嬪潎绾?1.50銆傝缁撴灉瀹氫綅鐨勬槸鍏变韩鍒濆鍖栧帇鍔涳紝涓嶆槸 RGD 鐗规湁缂洪櫡锛屼篃涓嶆敮鎸佽法璁剧疆缁熶竴浼樺娍銆俓n
## 6. 鏁版嵁銆侀厤缃笌澶嶇幇鍏ュ彛

浜嬪疄鏉ユ簮锛歕n
- fresh holdout锛歚results/tvt_revision_round6/mechanism_holdout/fresh_fast/`
- rollout 鍒嗘瀽锛歚results/tvt_revision_round6/release_rollout_analysis/`
- runtime 閰嶇疆锛歚config.yaml`
- 閿佸畾璇佹嵁鍗忚锛歚formal_protocol.yaml`
- 璁烘枃锛歚paper/main.tex` 涓?`paper/main.pdf`

澶嶇幇涓绘満鍒跺垎鏋愶細

```powershell
python tools/analyze_release_state_rollouts.py `
  --fast-root results/tvt_revision_round6/mechanism_holdout/fresh_fast/always_fast `
  --output-dir results/tvt_revision_round6/release_rollout_analysis `
  --seed-start 160 --seed-end 189 `
  --horizon 20 --gamma .99 --epsilon .02
```

鍥?3锛歕n
```powershell
python tools/generate_post_latency_evidence_figure.py `
  --rollout results/tvt_revision_round6/release_rollout_analysis/release_rollout_summary.csv `
  --output-dir paper/figures
```

鏈€缁堢粨鏋滀笉寰椾娇鐢?`den`锛涙€ц兘琛ㄤ繚鐣?Dis./Coll./Suc. 绛夊彲鐩存帴瑙ｉ噴鐨勬寚鏍囥€備互涓嬬洰褰曚粎淇濈暀浣滃巻鍙插璁★紝涓嶅緱杩涘叆姝ｆ枃鎴栧尶鍚嶅埗鍝侊細

- `results/tvt_revision_round5/invalid_precalibration_main_bridge_20260716/`
- `results/tvt_revision_round5/invalid_precalibration_transfer_bridge_20260716/`
