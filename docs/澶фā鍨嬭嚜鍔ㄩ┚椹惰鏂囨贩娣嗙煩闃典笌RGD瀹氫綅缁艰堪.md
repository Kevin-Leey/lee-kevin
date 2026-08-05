# 澶фā鍨嬭嚜鍔ㄩ┚椹舵枃鐚畾浣嶄笌 RGD 宸紓鐭╅樀

鏇存柊鏃ユ湡锛?026-07-17

## 1. 鏂囨。鐢ㄩ€擻n
鏈枃妗ｅ彧鏈嶅姟浜庡綋鍓?TVT 绋夸欢鐨勫畾浣嶃€丷elated Work 缁勭粐鍜屾湳璇竴鑷存€ф鏌ャ€傚畠涓嶆槸绯荤粺缁艰堪锛屼篃涓嶄互鏂囩尞鏁伴噺鏇夸唬閫愮瘒鏍搁獙銆傛渶缁堝紩鏂囬泦鍚堜互 `paper/references.bib` 鍜?`paper/main.tex` 鐨勫疄闄呭紩鐢ㄤ负鍑嗐€俓n
褰撳墠璁烘枃鍙涓€涓棶棰橈細

> 褰撳墠鐘舵€佸€煎緱璋冪敤鎱㈡帹鐞嗭紝涓嶇瓑浜庢參鎺ㄧ悊杩斿洖鏃朵粛鏈夊彲鎵ц鐨勭籂姝ｆ満浼氥€俓n
璁烘枃鎹鎶?**post-latency recoverability** 浣滀负杞﹁締渚?test-time computation allocation 鐨勬牳蹇冨彉閲忋€傜爺绌跺璞′笉鏄洿寮虹殑 driving backbone锛屼篃涓嶆槸璁╁ぇ妯″瀷鐩存帴鎺ョ杞﹁締銆俓n
## 2. 鏈€缁堟柟娉曞彛寰刓n
### 2.1 R-VoD锛氱悊璁哄璞n
Recoverable Value of Deliberation锛圧-VoD锛夊畾涔夊湪鎱㈣姹傜殑 release state銆傚彧鏈夊綋璇ョ姸鎬佷粛瀛樺湪涓€涓粡杩囧叡浜畨鍏ㄦ槧灏勫悗銆佺浉瀵?matched fast continuation 杈惧埌鐩爣澧炵泭鐨勫悎娉曞姩浣滄椂锛屾參鎺ㄧ悊鎵嶄繚鐣欑籂姝ｆ満浼氥€俓n
R-VoD 鏄湁闄愭椂鍩熺殑 oracle object銆傚綋鍓嶅疄楠屾病鏈夎娴嬪畬鏁?oracle membership锛屼篃娌℃湁璁粌 oracle estimator銆傚洜姝や笉寰楁妸 operational proxy 鍐欐垚 oracle calibration銆乷racle prediction 鎴栨櫘閫?value-of-computation 瀹氱悊銆俓n
### 2.2 RGD锛氬彲瀹炵幇鍒嗛厤鍣╘n
Recoverability-Gated Deliberation锛圧GD锛夊湪璇锋眰鍙戝嚭鍓嶈绠椾竴涓伐绋嬪寲 proxy锛歕n
- latency survival锛沑n- legal non-fast alternatives锛沑n- recovery-cost headroom锛沑n- 鍦ㄩ€氳繃 opportunity floor 鍚庣敤浜庢帓搴忕殑 present-need term銆俓n
RGD 鍙喅瀹氭槸鍚﹁喘涔版參璇锋眰銆傚畠涓嶇敓鎴愭柊鐨?driving backbone锛屼笉鏇挎崲 complete fast policy锛屼篃涓嶈幏寰楁渶缁?safety authority銆傛參璇锋眰 pending 鏈熼棿 fast policy 鎸佺画鎵ц锛涜繑鍥炲悗锛宲roposal 杩樺繀椤婚€氳繃 release-state legality 鍜屽叡浜?safety map銆俓n
### 2.3 璇佹嵁鍚堝悓

璁烘枃鎶婁互涓嬪璞″垎寮€璁板綍锛歲uery decision銆乺eturn survival銆乺elease divergence銆乺elease legality銆乻afety rewrite 鍜?final action銆傝 query-to-actuation contract 闃叉鎶娾€滃彂鍑鸿姹傗€濃€滄ā鍨嬭繑鍥炰笉鍚屽姩浣溾€濆拰鈥滃姩浣滄渶缁堣幏寰楁墽琛屾潈鈥濇贩涓轰竴璋堛€俓n
## 3. 鏂囩尞绨囦笌鐪熷疄宸紓

| 鏂囩尞绨?| 鍏稿瀷鐮旂┒闂 | 甯歌鍐崇瓥涓讳綋 | 涓?RGD 鐨勫叧绯?| 褰撳墠绋夸欢鐨勫尯鍒?|
| --- | --- | --- | --- | --- |
| LLM/VLM driving agents | 濡備綍鍒╃敤璇█鎴栬瑙夎瑷€妯″瀷鎻愬崌鐞嗚В銆佽鍒掍笌鎺у埗 | 澶фā鍨?driver 鎴栨牳蹇?planner | 鎻愪緵鍙璋冪敤鐨勯珮瀹归噺鎱㈠垎鏀?| RGD 鐮旂┒浣曟椂璐拱璇ュ垎鏀紝涓嶅０绉版敼杩涘叾鐢熸垚鑳藉姏 |
| World models and generative planners | 濡備綍琛ㄧず鏈潵銆佺敓鎴愬満鏅垨鎵╁睍鍊欓€夎建杩?| world model銆乨iffusion/generative planner | 澧炲姞 proposal space 鎴?rollout quality | RGD 涓嶆墿澶у€欓€夌┖闂达紝鍙垽鏂欢杩熷悗鏄惁浠嶅彲鑳界籂姝?|
| Dual-process and adaptive reasoning | 浣曟椂鏍规嵁鍙嶉銆佷笉纭畾鎬ф垨澶嶆潅搴﹀惎鐢ㄦ參鎬濊€?| fast-slow controller 鎴?learned router | 涓?RGD 鏈€鎺ヨ繎 | 鐜版湁瑙﹀彂閲忎富瑕佹弿杩?current salience锛汻GD 澧炲姞 release-state opportunity 绾︽潫 |
| Async and resource scheduling | 璇锋眰鑳藉惁鍦ㄩ槦鍒椼€佹椂闄愭垨棰勭畻鍐呰繑鍥炲拰澶嶇敤 | scheduler銆乹ueue manager | 澶勭悊璁＄畻鍙湇鍔℃€у拰鏃堕棿鎴愭湰 | RGD 杩涗竴姝ラ棶杩斿洖鍔ㄤ綔鏄惁杩樹繚鐣欒溅杈嗘帶鍒舵潈 |
| Viability and runtime assurance | 鍝簺鍔ㄤ綔鎴栨帶鍒跺櫒鍙畨鍏ㄦ墽琛?| safety filter銆乿iability set銆乺untime assurance | 鎻愪緵鏈€缁堟墽琛岃竟鐣?| RGD 涓嶆浛鎹㈠畨鍏ㄥ眰锛涘畠鍦ㄨ喘涔板墠浼拌寤惰繜鍚庢槸鍚﹀彲鑳藉瓨鍦ㄥ悎娉曠籂姝ｆЫ浣?|

## 4. 涓庨噸鐐瑰弬鑰冨伐浣滅殑鍏崇郴

### 4.1 澶фā鍨嬩笌瑙嗚璇█椹鹃┒

DiLu銆丏riveGPT4銆丩MDrive銆乂LM-Driver 浠ュ強 TVT 涓殑 driving-knowledge 鍜?vision-language navigation 宸ヤ綔锛屾牳蹇冭础鐚湪浜庡寮哄満鏅悊瑙ｃ€佺煡璇嗗埄鐢ㄣ€佸喅绛栫敓鎴愭垨闂幆椹鹃┒鑳藉姏銆傚畠浠鏄庨珮瀹归噺妯″瀷鑳藉浜х敓鏈変环鍊肩殑 proposal锛屼絾娌℃湁鐩存帴鍥炵瓟涓€涓凡璐拱鐨?proposal 鍦ㄨ繑鍥炴椂鏄惁浠嶈兘鏀瑰彉杞﹁締杩愬姩銆俓n
鍥犳锛屾湰绋夸笉鑳芥妸鈥滀娇鐢?Qwen3-8B鈥濆啓鎴愪富瑕佸垱鏂般€傝妯″瀷鏄叡浜?slow executor锛涗富瑕佸垱鏂版槸涓?executor 瑙ｈ€︾殑 vehicle-side allocator銆俓n
### 4.2 World model 涓庣敓鎴愬紡瑙勫垝

DriveDreamer銆丟enAD銆丏REWM銆丏iffusionDrive 鍜?C-TRAIL 绛夊伐浣滄墿灞曚簡鏈潵寤烘ā銆佸€欓€夎建杩圭敓鎴愩€佸父璇嗘绱㈡垨 trust-guided planning銆傚畠浠富瑕佹敼鍠勨€滄參鍒嗘敮鑳戒骇鐢熶粈涔堚€濄€俁-VoD/RGD 鐮旂┒涓嶅悓闂锛氣€滃綋缁撴灉杩斿洖鏃讹紝杩樻湁娌℃湁涓€涓浉瀵?complete fast continuation 鐨勫悎娉曠籂姝ｆ満浼氣€濄€俓n
涓ょ被璐＄尞鍙互缁勫悎锛屼絾涓嶈兘浜掔浉鏇夸唬銆傛洿寮虹殑 proposal generator 鍙兘鎻愰珮鏈轰細瀛樺湪鏉′欢涓嬪懡涓籂姝ｉ泦鍚堢殑姒傜巼锛屼笉鑳芥仮澶嶅凡缁忓洜寤惰繜娑堝け鐨勬満浼氥€俓n
### 4.3 鍙岃繃绋嬩笌寮傛鎺ㄧ悊

CogniDrive銆丩eapAD銆丩eapVAD銆丗ASIONAD銆丄daDrive 鍜?ThinkDrive 璇存槑鍙嶉銆乽ncertainty銆乧omplexity銆乴earned reward銆佸紓姝ラ槦鍒椾笌鐩爣閲嶉敋瀹氬彲浠ユ敼鍠?fast-slow cooperation銆侫syncDriver 鍒欏己璋冩參璇█鎸囧鐨勫紓姝ュ鐢ㄣ€俓n
鏈涓庤繖浜涘伐浣滅殑宸紓蹇呴』鍐欐垚鍙橀噺灞傞潰鐨勫尯鍒細

- current difficulty銆乽ncertainty 鎴?hazard 璇存槑鐜板湪鏄惁鍊煎緱娉ㄦ剰锛沑n- queue occupancy銆乨eadline 鎴?cache 璇存槑璇锋眰鏄惁鍙湇鍔★紱
- goal reanchoring 璇存槑杩斿洖缁撴灉濡備綍淇锛沑n- post-latency recoverability 璇存槑鍦ㄥ彂鍑鸿姹傚墠锛屽欢杩熷悗鐨勫悎娉曠籂姝ｆ満浼氭槸鍚﹀彲鑳戒粛瀛樺湪銆俓n
### 4.4 Metareasoning銆乿iability 涓庡畨鍏ㄤ繚璇乗n
缁忓吀 metareasoning 鍜?deliberation scheduling 宸ヤ綔鎻愪緵 computation cost銆乨eadline 鍜?expected value 鐨勭悊璁鸿儗鏅紱anytime computation 寮鸿皟浠峰€煎彇鍐充簬涓柇鏃堕棿銆俈iability theory銆乺untime assurance銆丼implex 绫绘灦鏋勫拰 RSS 鍒欐彁渚涘姩鎬佺幆澧冧腑鐨勫悎娉曟墽琛岃竟鐣屻€俓n
R-VoD 鏄潰鍚戦棴鐜溅杈嗗姩浣滅殑蹇呰鏈轰細瀵硅薄锛屼笉鏄柊鐨勯€氱敤 metareasoning 鐞嗚鎴?safety theorem銆傚綋鍓?Proposition 鍙湪鏄庣‘鐨?finite-horizon銆乶ested-feasible-set 鍜?non-increasing-advantage 鏉′欢涓嬫垚绔嬨€俓n
## 5. 褰撳墠璁烘枃鐨勪笁椤硅础鐚甛n
1. **Allocation object**锛氬畾涔?release-state 鐨?post-latency corrective opportunity锛屽苟缁欏嚭绌洪泦鎷掔粷鏉′欢鍜屾潯浠舵€?latency-erosion 鍛介銆俓n2. **Operational allocator**锛氱粰鍑虹敱 latency survival銆乴egal alternatives銆乺ecovery headroom 鍜?need ranking 缁勬垚鐨?RGD gate锛屽悓鏃朵繚鎸?backbone 涓?safety authority 涓嶅彉銆俓n3. **Stage-resolved evidence**锛氱敤 common Fast-only trajectories銆乧ontrolled latency erosion銆乧losed-loop endpoint 鍜?query-lifecycle audit 妫€楠?query placement銆佹満浼氫繚鎸佷笌鏈€缁堟墽琛屾潈銆俓n
涓嶅緱鎶婁互涓嬪唴瀹规媶鎴愰澶栤€滃垱鏂版ā鍧椻€濓細hidden `SLOWER` bridge銆乭ighway pass rule銆乼race schema銆乧ollision audit銆佸浘琛ㄧ敓鎴愯剼鏈垨鍖垮悕鍒跺搧鎵撳寘宸ュ叿銆傚畠浠睘浜庡疄鐜颁慨澶嶃€佸崗璁畬鏁存€ф垨璇佹嵁鍩虹璁炬柦銆俓n
## 6. 鏍稿績璇佹嵁涓庣粨璁哄己搴n
| 闂 | 璇佹嵁 | 鍏佽鐨勭粨璁?|
| --- | --- | --- |
| RGD 鏄惁閫夋嫨鏇村彲鎸佺画鐨?query state | 1.7 s matched Fast-only trajectories锛?4/111 瀵?23/100锛沺aired difference 0.256 [0.136, 0.375] | RGD-selected states 鏇村父淇濈暀 operational joint opportunity |
| 寤惰繜鏄惁渚佃殌鏈轰細 | 鍥哄畾 trajectory 鍜?route rule锛?.615銆?.486銆?.383 瀵瑰簲 0.7銆?.7銆?.7 s | 鍦ㄥ綋鍓嶅祵濂楁潯浠跺拰 highway protocol 涓嬶紝鏈轰細闅忓欢杩熶笅闄?|
| 瀹屾暣绯荤粺鏄惁淇濇寔杩愯 | RGD-only closed-loop sweep锛?--1.7 s 涓?29/30锛?.7 s 涓?27/30 | 缁欏嚭 bounded latency-stress profile锛屼笉璇嗗埆 allocator-latency interaction |
| 涓?endpoint 鏄惁浼樹簬 baselines | main seeds锛歊GD 28/30锛汿TC-risk 27/30锛涘叾浣?26/30 | 鏂瑰悜鏈夊埄浣嗙粺璁℃湭瑙ｅ喅锛屼笉寰楀绉?completion superiority |
| 4/5/6 杞﹂亾鑳藉惁鎵ц | density 2/3銆?40 episodes銆侀浂闄勫姞寤惰繜 | 楠岃瘉璁剧疆鎵ц鍜?traffic-stress sensitivity锛屼笉鏀寔缁熶竴璺ㄨ缃紭鍔?|

## 7. Related Work 鍐欎綔椤哄簭

1. 鍏堣鏄庡ぇ妯″瀷 driving銆亀orld model 鍜?generative planner 鎻愰珮 proposal capability銆俓n2. 鍐嶈鏄?dual-process銆乺isk/uncertainty routing 鍜?asynchronous systems 绠＄悊 activation銆乧adence銆乹ueue 鎴?return repair銆俓n3. 鎺ョ潃寮曞叆 metareasoning銆乤nytime computation銆乿iability 鍜?runtime assurance锛屽缓绔嬧€滆绠楁湡闂寸姸鎬佺户缁紨鍖栤€濈殑鐞嗚鑳屾櫙銆俓n4. 鏈€鍚庢彁鍑虹己鍙ｏ細鐜版湁鍙橀噺娌℃湁鍦ㄨ喘涔板墠鐩存帴琛ㄧず delayed release state 鏄惁浠嶄繚鐣?matched-fast-relative corrective opportunity銆俓n
璇ラ『搴忎娇鐮旂┒鏁呬簨浠庘€滆兘鐢熸垚浠€涔堚€濊浆鍒扳€滀綍鏃朵粛鏈夋潈鎵ц鈥濓紝鑰屼笉鏄妸璁烘枃鍐欐垚妯″潡娓呭崟銆俓n
## 8. 绂佹鎭㈠鐨勬棫鍙ｅ緞

浠ヤ笅鏈鎴栦富寮犱笌鏈€缁堣鏂囦笉涓€鑷达紝涓嶅簲閲嶆柊鍐欏叆姝ｆ枃銆佹憳瑕併€佸浘娉ㄦ垨閰嶇疆璇存槑锛歕n
- `ASRO` 鎴?`ASRO-conditioned counterfactual deliberation`锛沑n- 鎶?RGD 鍐欐垚 oracle estimator銆乻afety certificate 鎴?universal value-of-computation锛沑n- `compute-matched superiority`銆乣completion superiority` 鎴栬法 simulator generalization锛沑n- 鎶?query count銆乻low-fast disagreement 鎴?safety override 鍗曠嫭褰撲綔鏈夋晥骞查锛沑n- 鎶?density 3.0 鐨勯珮纰版挒鐜囬殣鍘绘垨瑙ｉ噴鎴?RGD 鐗规湁缂洪櫡锛沑n- 鎶?1.7 s 鍐欐垚鐪熷疄纭欢鎺ㄧ悊寤惰繜銆俓n
## 9. 鍗曚竴浜嬪疄婧怽n
- 杩愯榛樿鍊硷細`config.yaml`锛沑n- 姝ｅ紡鍗忚涓?submission contract锛歚formal_protocol.yaml`锛沑n- 鍙鐜板疄楠岀粨鏋滐細`results/tvt_revision_round5/`锛沑n- 鏈€缁堣鏂囷細`paper/main.tex` 涓?`paper/main.pdf`锛沑n- 璁烘枃浜嬪疄瀹¤锛歚tools/audit_tvt_manuscript.py`锛沑n- 鍖垮悕鍒跺搧楠岃瘉锛歚tools/verify_tvt_anonymized_artifact.py`銆俓n
浠讳綍鏁版嵁銆乻eed銆侀槇鍊笺€佸欢杩熸垨鏈鍙戠敓鍙樺寲鏃讹紝蹇呴』鍚屾鏇存柊涓婅堪浜嬪疄婧愬強鍥涗唤鏍稿績 TVT 鏂囨。锛屼笉鑳藉彧鏀硅鏂囪〃鏍兼垨鍙敼閰嶇疆銆俓n