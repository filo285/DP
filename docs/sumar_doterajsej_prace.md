# Súhrn doterajšej práce

Stav overený k 16. októbru 2026.

**Cieľ projektu**

Pripravujeme výber fundusových obrázkov na anotáciu pomocou aktívneho učenia. Model segmentuje optický disk a jeho vnútornú exkaváciu (cup). Okrem segmentácie chceme zistiť, pri ktorých obrázkoch sú jeho predikcie nestabilné, a overiť, či anotovanie týchto obrázkov prinesie väčšie zlepšenie než náhodný výber pri rovnakom rozpočte anotácií.

**Čo máme implementované**

Máme dve samostatné metódy získavania opakovaných predikcií:

- **MC dropout:** jeden natrénovaný model spracuje obrázok opakovane. Pri vyhodnocovaní ponechávame náhodný dropout aktívny a batch normalization v režime vyhodnocovania. Pilot používa 30 prechodov na obrázok.
- **Deep ensembles:** niekoľko nezávisle trénovaných modelov spracuje rovnaký obrázok. Každý člen má vlastný seed, inicializáciu, optimalizátor a priebeh tréningu, pričom používa rovnaké rozdelenie dát. Implementovaný je tréning aj vyhodnotenie; predvolené nastavenie počíta s piatimi členmi. Pri inferencii je dropout vypnutý.

Kód metód je oddelený v `src/dp2/mc_dropout/` a `src/dp2/deep_ensembles/`. Spoločné výpočty pravdepodobností, geometrie a exportov sú v `src/dp2/common/`.

Pre obe metódy počítame priemerné pravdepodobnosti tried, výslednú segmentáciu, mapy štandardnej odchýlky pravdepodobností, prediktívnej entropie a BALD. Výsledná segmentácia vzniká výberom najpravdepodobnejšej triedy z priemerných pravdepodobností.

**Ako vyjadrujeme neistotu jedným číslom**

Ako hlavné skóre obrázka sme zvolili priemerný pixelový BALD v predikovanej oblasti disku vrátane cupu:

`BALD = H(priemer pravdepodobností) − priemer H(jednotlivých predikcií)`

Entropia priemernej predikcie vyjadruje, nakoľko je výsledné rozdelenie pravdepodobností nejednoznačné. BALD navyše zohľadňuje, či sa jednotlivé predikcie navzájom nezhodujú. Štandardná odchýlka pravdepodobností opisuje ich kolísanie medzi prechodmi alebo modelmi.

Pri deep ensembles ide o skóre nezhody rovnakého matematického tvaru ako BALD; nezávislé modely nie sú presnými vzorkami z bayesovského posterioru.

Pri priemerovaní neistoty obrázka **vynechávame predikované pixely pozadia a doplnené okraje obrázka**. Trieda pozadia zostáva súčasťou pravdepodobnostného rozdelenia, takže zachovávame aj neistotu na hranici objektu. Skóre nepotrebuje referenčnú anotačnú masku. Pri prázdnej predikovanej oblasti skóre označíme ako neplatné a obrázok vyradíme z číselného rebríčka.

BALD je naša východisková voľba pre výber obrázkov. Zatiaľ sme experimentálne nepreukázali, že bude pre naše dáta najlepšou stratégiou. Entropiu a štandardnú odchýlku ponechávame na porovnanie a analýzu.

**Neistota anatomického merania vCDR**

Implementovali sme aj vertikálny pomer cup-to-disc:

`vCDR = vertikálna výška cupu / vertikálna výška celého disku`

Celý disk zahŕňa rim aj cup. Používame pomer výšok, nie plôch. vCDR vypočítame samostatne z každej predikcie a ako neistotu obrázka použijeme populačnú štandardnú odchýlku týchto pomerov, s menovateľom N.

Takto rozlišujeme nezhodu v segmentácii pomocou BALD a nestabilitu konkrétneho anatomického merania pomocou vCDR. Skóre ukladáme a zoraďujeme samostatne; nesčítavame ich.

Ak cup chýba, ale disk existuje, vCDR je 0 a prípad sa osobitne zaznamená. Ak chýba disk, pomer nie je definovaný. Ak sa to stane v ktoromkoľvek prechode, výsledná neistota vCDR je neplatná. Nulová štandardná odchýlka sama osebe nepotvrdzuje správnosť segmentácie: všetky predikcie môžu robiť rovnakú chybu.

**Čo sme overili a čo ukázali výstupy**

Pôvodný model sme overili na všetkých 400 validačných obrázkoch. Uložený report potvrdzuje zhodu s metrikami pôvodného checkpointu: Dice disku približne 0,946, Dice cupu 0,873 a priemerná absolútna chyba vCDR 0,054.

MC dropout pilot zahŕňa osem obrázkov s 30 prechodmi. BALD označil ako najneistejší obrázok **V0001**, zatiaľ čo štandardná odchýlka vCDR označila **V0006**. Obe skóre mali rovnaké štyri obrázky na vrchole rebríčka, ale v inom poradí. Rozdiel je očakávateľný: zmena lokálnej segmentácie a zmena pomeru vertikálnych výšok nemusia byť rovnako výrazné. Pri V0006 v jednom prechode zmizol cup, čo prispelo k nestabilite vCDR.

V0004 mal zároveň slabšiu segmentáciu cupu napriek pomerne nízkej neistote. Pilot preto ukazuje, že neistota nie je priamou mierou chyby; vzťah treba vyhodnotiť na väčšej vzorke.

Pri deep ensembles sme zatiaľ spustili iba krátke overenie funkčnosti: dva modely, jedna epocha, osem tréningových a štyri validačné obrázky pri rozlíšení 128 × 128. V jeho výstupoch bolo vCDR 1 a jeho štandardná odchýlka 0, pretože chybné masky cupu aj disku mali vertikálny rozsah celého obrázka. Ide o dôsledok nedostatočne natrénovaných modelov, nie o dôkaz spoľahlivosti. Riadny tréning ensemble a férové porovnanie s MC dropout ešte zostávajú.

Máme exporty máp a segmentácií, tabuľky skóre, samostatné rebríčky BALD a vCDR, vCDR pre každý prechod alebo model a reporty nastavení. Pre V0001 a V0006 sme pripravili aj spoločný vizuálny prehľad. Doplnili sme testy výpočtov, geometrie, správania modelov a kontroly checkpointov.

**Čo sme zistili z literatúry o množinách dát**

- **Gal a kol. (2017):** v hlavnom MNIST experimente používajú samostatnú validáciu so 100 obrázkami na nastavenie regularizácie a štandardný test s 10 000 obrázkami. V inom porovnaní používajú 5 000 validačných obrázkov. Rozdelenie treba interpretovať v kontexte konkrétneho experimentu.
- **Hemelings a kol. (2020):** jasne oddeľujú tréning (4 935 obrázkov), validáciu (679) a test (1 424). Rozdelenie robia podľa pacientov. Validácia slúži na výber modelu a zastavenie tréningu, test na konečné hodnotenie. Je to najjasnejší overený vzor pre náš protokol s oboma hodnotiacimi množinami.
- **SEQUAL (2026):** začínajú so 150 označenými výrezmi, poolom 1 875 výrezov a testom 1 188 výrezov. Štyri výbery po 200 výrezoch rozšíria tréning na 950. Testovacie obrázky zostávajú počas AL cyklu rovnaké. Samostatnú validáciu v hlavnom rozdelení neuvádzajú; zastavenie tréningu určujú podľa priebehu tréningového mIoU. Majú aj externé testovanie. Priradenie všetkých výrezov jedného pacienta do jednej množiny však nie je v opise rozdelenia výslovne doložené.
- **WSI-BayesUNet:** uvádza úvodný tréning, vyhradenú validáciu a pool. Samostatný test nepopisuje a formulácia o poole zo všetkých pacientov necháva oddelenie pacientov nejasné.
- **RLDAL:** overili sme verejné časti článku a oficiálny kód vrátane uložených rozdelení a skriptu so seedom. Plný experimentálny protokol bol neprístupný, preto nevieme úplne posúdiť rozdelenia všetkých jeho datasetov.

Počas jedného AL behu sa rozširuje označená časť tréningu a zmenšuje pool. Test a prípadná validácia zostávajú oddelené. Medzi nezávislými opakovaniami experimentu sa rozdelenie môže zmeniť, ak to protokol stanovuje. Pri simulovanom AL už anotácie existujú, ale algoritmus ich smie použiť až po výbere príslušného obrázka.

**Aktuálny stav a ďalší postup**

Máme implementovaný a pilotne overený výpočet neistoty pre obe metódy. **Kompletný cyklus active learningu s výberom, rozšírením tréningu a opakovaným tréningom zatiaľ implementovaný nie je.** Súčasné rebríčky vznikajú na validačných dátach na účely analýzy; validačné obrázky nepresúvame do tréningu.

Uložené rozdelenie obsahuje 1 050 tréningových, 400 validačných a 400 testovacích obrázkov. Pre AL treba vopred stanoviť počiatočnú označenú podmnožinu tréningu a zostávajúci pool, veľkosť výberu, rozpočet a opakovania. Validačné aj testovacie obrázky majú zostať pevné a pri dostupných identifikátoroch treba overiť oddelenie pacientov.

Ďalšie kroky sú riadne natrénovať deep ensembles, rozšíriť analýzu neistoty, implementovať AL cyklus a porovnať náhodný výber, BALD a neistotu vCDR pri rovnakom rozpočte. Obmedzený tréning nesmie začínať z checkpointu, ktorý už videl celý tréningový pool. Prínos stratégie bude potrebné preukázať na krivkách kvality segmentácie a chyby vCDR podľa počtu anotovaných obrázkov.

**Zdroje**

- Gal a kol.: https://proceedings.mlr.press/v70/gal17a/gal17a.pdf
- Hemelings a kol.: https://onlinelibrary.wiley.com/doi/full/10.1111/aos.14193
- SEQUAL: https://discovery.ucl.ac.uk/id/eprint/10225571/
- WSI-BayesUNet: https://proceedings.mlr.press/v316/cui26a.html
- RLDAL: https://www.sciencedirect.com/science/article/pii/S0957417426035712
- RLDAL, oficiálny kód: https://github.com/diaodiao317/RLDAL

Podklady v projekte: `README.md`, `docs/image_uncertainty.md`, implementácia v `src/dp2/` a uložené reporty a tabuľky v `runs/mc_dropout/my_test_01/` a `runs/deep_ensembles/my_test_01/`.
