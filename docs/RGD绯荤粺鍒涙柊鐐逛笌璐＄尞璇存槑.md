# RGD 绯荤粺鍒涙柊鐐广€佽瘉鎹笌澹版槑杈圭晫

鏇存柊鏃ユ湡锛?026-07-17

鏈枃妗ｅ彧鎻忚堪褰撳墠 IEEE TVT 鐗堟湰銆傛渶缁堣鏂囦綅浜?`paper/main.tex` 鍜孿n`paper/main.pdf`锛涙寮忚繍琛屽崗璁綅浜?`formal_protocol.yaml`銆備慨澶嶅墠缁撴灉銆乗n鏃?`paper2` 璺緞鍜屾湭杩涘叆姝ｆ枃鐨勬帰绱㈡€у疄楠屽潎涓嶆瀯鎴愯鏂囪瘉鎹€俓n
## 1. 鏍稿績绉戠爺闂

褰撳墠鍦烘櫙鍥伴毦锛屽苟涓嶆剰鍛崇潃鎱㈡帹鐞嗚繑鍥炴椂浠嶆湁鍙墽琛岀殑绾犳鏈轰細銆傛參妯″瀷鍦ㄦ帹鐞嗘湡闂达紝
杞﹁締鍜屼氦閫氱姸鎬佺户缁紨鍖栵紝鍚堟硶鍔ㄤ綔銆佽溅璺濆拰瀹夊叏灞傛潈闄愬彲鑳藉彂鐢熷彉鍖栥€傚洜姝わ紝鎱㈡帹鐞哱n鍒嗛厤涓嶈兘鍙敱 query-state risk銆乽ncertainty 鎴?complexity 鍐冲畾锛岃繕蹇呴』鑰冭檻
**post-latency recoverability**锛氳繑鍥炵姸鎬佹槸鍚︿粛淇濈暀鐩稿 matched fast continuation
鏈夋剰涔夌殑鍚堟硶绾犳鍔ㄤ綔銆俓n
璇ラ棶棰樺皢鏈枃涓庝互涓嬪伐浣滃尯鍒嗗紑锛歕n
- proposal-capability 鏂规硶鍥炵瓟鎱㈠垎鏀兘澶熸彁鍑轰粈涔堬紱
- risk/uncertainty routing 鍥炵瓟褰撳墠鐘舵€佹槸鍚﹀€煎緱鏇村鎺ㄧ悊锛沑n- queue/resource scheduling 鍥炵瓟璇锋眰鑳藉惁鍦ㄨ祫婧愮害鏉熷唴瀹屾垚锛沑n- post-return reanchoring 鍥炵瓟杩斿洖鐩爣濡備綍涓庡綋鍓嶇姸鎬侀噸鏂板榻愶紱
- RGD 鍥炵瓟鍦ㄥ彂璧疯姹傚墠锛屽欢杩熷悗鏄惁浠嶅彲鑳戒繚鐣欑籂姝ｆ潈闄愩€俓n
## 2. 涓夐」鏍稿績璐＄尞

### 2.1 闂瀵硅薄锛歊ecoverable Value of Deliberation

Recoverable Value of Deliberation锛圧-VoD锛夋妸鎱㈡帹鐞嗘満浼氬畾涔夊湪 release state锛氬湪
鏈夐檺鏃跺煙銆佸叡浜畨鍏ㄦ槧灏勫拰 matched fast continuation 涓嬶紝鏄惁瀛樺湪杈惧埌鐩爣浼樺娍杈归檯
鐨?viable corrective action銆傜┖ corrective set 鏄嫆缁濇參璋冪敤鐨勫繀瑕佹潯浠躲€俓n
R-VoD 鏄悊璁?oracle object銆傚綋鍓嶅疄楠屾病鏈?oracle 鏍囩锛屼篃娌℃湁缁欐湁闄愭椂鍩熷洖鎶ュ弬鏁癨n璧嬬粡楠屽€硷紝鍥犳涓嶅緱鎶婂伐绋?proxy 鍐欐垚 oracle estimator銆佹鐜囨牎鍑嗗櫒鎴栧畨鍏ㄨ瘉鏄庛€俓n
### 2.2 鏂规硶锛歊ecoverability-Gated Deliberation

Recoverability-Gated Deliberation锛圧GD锛夋槸鍙璁＄殑 vehicle-side allocator銆傚畠鍦╘n鎱㈣緭鍑轰骇鐢熷墠璁＄畻锛歕n
- latency survival `L_t`锛沑n- legal-alternative fraction `A_t`锛沑n- recovery-cost headroom `H_t`锛沑n- operational opportunity `E_t = L_t sqrt(A_t H_t)`锛沑n- need score `G_t` 涓?priority `P_t = E_t G_t`銆俓n
鍙湁 `E_t >= 0.20`銆乣P_t >= 0.16`銆佽嚦灏戜竴涓悎娉曢潪 fast alternative銆侀绠楀彲鐢ㄣ€乗ncooldown 缁撴潫涓?executor/latency provenance 鍙敤鏃舵墠璐拱鎱㈣皟鐢ㄣ€俽isk 鍜?junction
pre-screen 鍙兘鎺掑簭锛屼笉鑳界粫杩?opportunity floor銆俓n
### 2.3 璇佹嵁鍗忚锛氫粠 query 鍒?actuation 鐨勬潈闄愬璁n
鎵€鏈夊湪绾?allocator 鍏变韩 fast controller銆丵wen3-8B slow executor銆乹uery map銆佸姩浣淺n鎺ュ彛銆侀绠椼€乧ooldown 鍜?downstream safety map銆傛參璋冪敤鏈熼棿 complete fast policy
閫愭閲嶇畻锛涜繑鍥炲悗鍐嶆妫€鏌ュ綋鍓嶅悎娉曞姩浣滀笌瀹夊叏鏄犲皠銆傛棩蹇楀尯鍒嗭細

- purchased query 涓?in-horizon return锛沑n- raw-to-queue rewrite锛沑n- release-state unavailability锛沑n- release divergence锛沑n- downstream rewrite锛沑n- preserved final authority銆俓n
杩欎娇璁烘枃鑳藉鍥炵瓟鎱㈣绠楀湪鍝噷澶卞幓鏉冮檺锛岃€屼笉鏄妸姣忔璋冪敤閮界畻浣滄湁鏁堝共棰勩€俓n
## 3. 褰撳墠閿佸畾杩愯閰嶇疆

| 椤圭洰 | 鏈€缁堝€?|
|---|---|
| 涓荤幆澧?| `highway-v0`锛? lanes锛宒ensity parameter 2.0锛?0 vehicles |
| 鏃跺煙涓庨鐜?| 30 s锛宲olicy/simulation 10 Hz |
| 鍒濆/鐩爣閫熷害 | 26 m/s锛?0--30 m/s锛? 涓洰鏍囨。浣?|
| 涓诲欢杩?| 1.7 s锛屽搴?17 policy steps |
| 鎱㈣皟鐢ㄩ绠?cooldown | 6 / 20 frames |
| RGD threshold/floor | 0.16 / 0.20 |
| comparator calibration | Random 0.02锛沀ncertainty cutoff 1.00 + exposure 0.07锛汿TC 0.43 |
| 鎱㈡墽琛屽櫒 | `Qwen/Qwen3-8B`锛宼emperature 0锛?4 output tokens锛?20 s timeout |
| 鏀寔缁勪欢 | memory銆乫ew-shot銆乼race-cache 鍧囧叧闂?|
| 鍔ㄤ綔妗?| risk-scoped hidden `SLOWER` bridge 鍚敤 |
| safety | unified arbitration锛況elease-state legality + shared safety map |

API 鍑嵁涓嶅啓鍏?`config.yaml`锛屽繀椤婚€氳繃鐜鍙橀噺鎻愪緵銆傛渶缁堝巻鍙茬粨鏋滀互姣忎釜 run 鐨刓n`runtime_manifest.json` 鍜?`experiment_snapshot.json` 涓轰簨瀹炴潵婧愶紱褰撳墠閰嶇疆鐢ㄤ簬鏈潵
澶嶈窇骞朵笌杩欎簺閿佸畾鍊间繚鎸佷竴鑷淬€俓n
## 4. 鏈€缁堣瘉鎹甛n
### 4.1 涓婚棴鐜粨鏋淺n
涓绘瘮杈冧娇鐢?seeds 100--129锛屾瘡涓?allocator 30 涓?episode锛歕n
| Allocator | Success | Collision | Route | Distance (m) |
|---|---:|---:|---:|---:|
| RGD | 28/30 | 0.067 | 0.951 | 596.27 |
| Fast-only | 26/30 | 0.133 | 0.896 | 559.77 |
| Random | 26/30 | 0.133 | 0.908 | 573.92 |
| Uncertainty | 26/30 | 0.133 | 0.896 | 561.03 |
| TTC-risk | 27/30 | 0.100 | 0.924 | 579.31 |

RGD 鐨勬柟鍚戞€х粨鏋滄洿濂斤紝浣?paired exact tests 缁?Holm 璋冩暣鍚庝笉鏀寔 completion
superiority銆傚洜姝や富绔偣鏄郴缁熶竴鑷存€ц瘉鎹紝涓嶆槸 SOTA 鎴栨樉钁椾紭瓒婃€у０鏄庛€俓n
### 4.2 鏍稿績鏈哄埗璇佹嵁

鍦ㄧ浉鍚?Fast-only 杞ㄨ抗涓婏紝1.7 s 鍚庝粛鍚屾椂婊¤冻 opportunity floor 鍜?legal
alternative 鐨勬瘮渚嬩负锛歕n
- RGD锛?4/111 = 0.486锛沑n- TTC-risk锛?3/100 = 0.230锛沑n- paired seed-bootstrap difference锛?.256锛?5% CI [0.136, 0.375]銆俓n
璇ョ粨鏋滈獙璇?query placement 鐨?operational-proxy persistence锛屼笉鏄?oracle 鏍囩銆乗nslow-output quality 鎴?endpoint causal effect銆俓n
鍥哄畾杞ㄨ抗鍜?route rule锛屼粎澧炲姞 replay delay 鏃讹紝RGD retained joint fraction 涓猴細

- 0.7 s锛?0/130 = 0.615锛沑n- 1.7 s锛?4/111 = 0.486锛沑n- 2.7 s锛?3/60 = 0.383銆俓n
杩欎笌 conditional latency erosion 涓€鑷达紝浣嗕笉鏋勬垚鏃犳潯浠跺崟璋冩€у畾鐞嗐€俓n
### 4.3 闂幆寤惰繜绔偣

seeds 130--159 鐨?RGD-only sweep 寰楀埌锛歕n
| Added delay | Distance (m) | Collision | Success |
|---:|---:|---:|---:|
| 0.0 s | 597.10 | 0.033 | 29/30 |
| 0.7 s | 594.77 | 0.033 | 29/30 |
| 1.7 s | 586.44 | 0.033 | 29/30 |
| 2.7 s | 563.26 | 0.100 | 27/30 |

璇ュ疄楠岀粰鍑?bounded latency-stress signature锛屼笉璇嗗埆 allocator--latency interaction銆俓n
### 4.4 闆跺欢杩熻溅閬?瀵嗗害杩佺Щ

4/5/6 lanes銆乨ensity parameter 2.0/3.0銆丷GD/TTC-risk/Fast-only銆乻eeds 0--29
鏋勬垚 540 涓浂闄勫姞寤惰繜 episode銆俁GD 鍏牸 Success 涓?30銆?1銆?9銆?6銆?6銆?6銆俓n
density 3.0 鐨勯珮纰版挒鍦ㄤ笁涓?allocator 涓悓姝ュ嚭鐜帮紝骞朵即闅忓垵濮?forward gap 绾︾缉灏廫n1.50 鍊嶃€傚洜姝よ鐜拌薄鏄?Highway-Env 鍒濆鍖栧帇鍔涜竟鐣岋紝涓嶆槸 RGD bridge 澶辨晥锛屼篃涓峔n鏀寔璺ㄨ缃?superiority銆俓n
## 5. 纰版挒鏍瑰洜涓庝慨澶峔n
鏃у疄楠屼腑锛孒ighway-Env 鍦ㄥ悕涔夌洰鏍囬€熷害杈惧埌 20 m/s 鍚庨殣钘?`SLOWER`锛屼娇 fast rule
瑕佹眰 emergency brake 鏃跺姩浣滃彲鑳介€€鍖栦负 `IDLE`銆傚綋鍓嶇郴缁熼€氳繃 risk-scoped hidden
slower bridge 鏆撮湶鍒跺姩鍔ㄤ綔骞舵槧灏勫埌鏇翠綆鐗╃悊閫熷害鐩爣銆俓n
淇鍚庯細

- 150 涓富瀹為獙锛歚emergency_idle_without_slower=0`锛沑n- 120 涓棴鐜欢杩熷疄楠岋細`emergency_idle_without_slower=0`锛沑n- 540 涓縼绉诲疄楠岋細bridge 鍚敤銆侀浂姝ｅ欢杩熴€佹湁鏁?lane index锛屼笖鏃犺鍔ㄤ綔閫€鍖栥€俓n
淇鍓嶇洰褰曞彧鑳界敤浜庢牴鍥犲璁★紝涓嶅緱杩涘叆璁烘枃琛ㄦ牸銆佺粺璁″垎鏋愭垨鍖垮悕鍒跺搧銆俓n
## 6. Claim--Evidence Map

| 澹版槑 | 璇佹嵁 | 杈圭晫 |
|---|---|---|
| post-latency recoverability 鏄嫭绔?allocation variable | release-state 瀹氫箟銆乺isk/recoverability mismatch銆乧ommon-trajectory comparison | 鏀寔 |
| RGD 鍙璁″苟婊¤冻 eligibility contract | 122 queries銆? invariant violation銆乹uery-to-release trace | 鏀寔 |
| 寤惰繜渚佃殌 operational opportunity | 0.7/1.7/2.7 s 鍥哄畾杞ㄨ抗瀹¤ | 鏉′欢鎬ф敮鎸?|
| RGD completion 鏄捐憲浼樹簬 baselines | paired endpoint 鏈樉钁?| 涓嶆敮鎸?|
| RGD 鏄畨鍏ㄨ瘉鏄庢垨 oracle estimator | 鏃?oracle labels/姝ｅ紡瀹夊叏璇佹槑 | 涓嶆敮鎸?|
| RGD 璺ㄥ瘑搴︾粺涓€鍗犱紭 | 540-run sweep 鏃犵粺涓€ superiority | 涓嶆敮鎸?|
| RGD 瓒呰秺 PADriver | 骞冲彴涓庢潯浠朵笉鍖归厤 | 涓嶆敮鎸侊紱浠呬綔鏍煎紡瀵归綈澶栭儴鍙傝€?|

## 7. 褰撳墠杩芥函璺緞

| 鍐呭 | 璺緞 |
|---|---|
| 鏈€缁堣鏂?| `paper/main.tex`, `paper/main.pdf` |
| 姝ｅ紡鍗忚 | `formal_protocol.yaml` |
| 榛樿澶嶈窇閰嶇疆 | `config.yaml` |
| RGD route owner | `dilu/driver_agent/reasoning/rgd_core.py` |
| hidden slower bridge | `dilu/runtime_support.py` |
| 涓诲垎鏋?| `results/tvt_revision_round5/main_analysis/` |
| 闂幆寤惰繜 | `results/tvt_revision_round5/latency_endpoint_recalibrated/` |
| 杞﹂亾/瀵嗗害杩佺Щ | `results/tvt_revision_round5/transfer_analysis_final/` |
| 纰版挒瀹¤ | `results/tvt_revision_round5/collision_audit_final_main/`, `collision_audit_final_latency/` |
| 瀹為獙璁捐 | `docs/RGD瀹為獙璁捐鏂规.md` |
| 鏈€鏂板瀹?| `docs/IEEE_TVT_淇鍚庡瀹℃剰瑙佷笌璇勫垎.md` |
| 鍖垮悕鍒跺搧 | `paper/tvt_anonymized_artifact_v1.zip` |

## 8. 鏈€缁堝畾浣峔n
RGD 涓嶆槸鏂扮殑鑷姩椹鹃┒澶фā鍨嬶紝涔熶笉鏄?safety controller銆傚畠鏄潰鍚戞參--蹇棴鐜┚椹剁殑
recoverability-conditioned computation allocator锛氬厛鍒ゆ柇寤惰繜鍚庢槸鍚︿粛鍙兘淇濈暀绾犳
鏉冮檺锛屽啀鍐冲畾鏄惁璐拱鎱㈡帹鐞嗐€傝鏂囩殑鏍稿績浠峰€煎湪浜庢彁鍑鸿繖涓€ allocation object銆佺粰鍑篭n鍙璁″疄鐜帮紝骞剁敤 matched trajectory銆乴atency erosion 鍜?query-to-actuation evidence
灏嗗叾涓庡綋鍓嶉闄╂垨鍥伴毦搴﹁Е鍙戞槑纭尯鍒嗐€俓n