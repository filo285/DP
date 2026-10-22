# MC dropout a deep ensembles

Výpočet skóre celého obrázka a vCDR z každého člena je popísaný v
[návode pre image uncertainty](../../../docs/image_uncertainty.md).
Primárne skóre je priemerný BALD v predikovanej oblasti disku; anatomické skóre
je štandardná odchýlka vCDR medzi členmi. Ukladajú sa samostatne.

## Kde je ktorý kód

```text
src/dp2/
  baseline.py                 spoločný model, dáta a metriky z DP1
  common/
    uncertainty.py            priemer, odchýlka, entropia a skóre
    export.py                 spoločný formát PNG, NPZ a CSV
  mc_dropout/
    inference.py              opakované prechody s aktívnym dropoutom
    cli.py                    kontrola baseline a výpočet MC dropout
  deep_ensembles/
    __main__.py               spoločný vstup pre train a evaluate
    README.md                 návod pre deep ensembles
    tests/                    testy iba pre deep ensembles
    train.py                  samostatný tréning každého člena
    checkpoints.py            načítanie a kontrola checkpointov
    inference.py              jeden deterministický prechod každého člena
    cli.py                    výpočet a export ensemble neistoty
  uncertainty.py              spätná kompatibilita starých importov
```

Spoločné matematické operácie sú úmyselne na jednom mieste, aby obe metódy
používali rovnaké definície skóre. Samotný tréning a inferencia sú oddelené.

## Plný experiment

V termináli otvorenom v koreňovom priečinku projektu:

```powershell
.\.venv\Scripts\python.exe src/dp2/deep_ensembles train --members 5 --output runs/deep_ensembles/full
.\.venv\Scripts\python.exe src/dp2/deep_ensembles evaluate --ensemble runs/deep_ensembles/full --limit 0 --output runs/deep_ensembles/full_predictions
```

Prvý príkaz natrénuje päť samostatných modelov so seedmi 42 až 46. Všetky majú
rovnaké dáta, architektúru a hyperparametre. Každý má vlastnú inicializáciu
dekodéra a hlavy, poradie obrázkov a augmentácie. Encoder štandardne začína
z rovnakých ImageNet váh ako v pôvodnom baseline; nejde o úplne náhodný encoder.
Jeho prvé načítanie môže vyžadovať internet. Parameter `--encoder-weights none`
zapne úplne náhodnú inicializáciu, čo však mení podmienky porovnania.

Každý člen sa učí najviac 60 epoch s early stoppingom po 10 epochách bez
zlepšenia. Ukladá sa checkpoint s najlepším priemerom validačného Dice pre disk
a pohárik. Modely sa trénujú postupne, takže pri tréningu nepotrebujú naraz
pamäť pre päť sietí. Pri inferencii sa všetci členovia načítajú naraz; podľa
potreby možno použiť `--device cpu`.

Druhý príkaz vypočíta neistotu na celej validačnej množine. Pri každom členovi
je dropout vypnutý. Výsledkom je priemer pravdepodobností, štandardná odchýlka,
prediktívna entropia a vzájomná informácia medzi členom a predikciou. Stĺpec
`bald` zachováva spoločný formát s MC dropoutom; pri ensemble označuje analogické
skóre nesúhlasu, nie presný výpočet z bayesovského posterioru.

## Rýchly technický test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src/dp2/deep_ensembles train --smoke --output runs/deep_ensembles/smoke_new
.\.venv\Scripts\python.exe src/dp2/deep_ensembles evaluate --ensemble runs/deep_ensembles/smoke_new --output runs/deep_ensembles/smoke_new_predictions
```

Smoke test používa dva modely, jednu epochu, 8 tréningových a 4 validačné
obrázky pri rozlíšení 128 × 128. Overuje funkčnosť tréningu, ukladania, načítania
a exportu. Jeho segmentácie nie sú použiteľné ako výskumné výsledky.

## Výstupy

- `runs/deep_ensembles/<beh>/ensemble.json`: nastavenia, seedy a kontrolné súčty.
- `member_00_seed_42/best.pth` a ďalšie priečinky: samostatné modely.
- `member_00_seed_42/history.json`: strata a validačné metriky počas tréningu.
- `splits/`: konkrétne rozdelenie obrázkov použité pri tréningu.
- Priečinok inferencie: `report.json`, `image_scores.csv`, číselné mapy `*.npz`
  a náhľady prvých ôsmich obrázkov `*.png`.

Existujúci výstupný priečinok sa neprepisuje. Pri novom pokuse zvoľ nový názov.
Obnovenie prerušeného tréningu zatiaľ nie je implementované.

MC dropout spustíš samostatne cez `scripts/run_mc_dropout.py`; jeho nové výstupy
štandardne smerujú do `runs/mc_dropout/pilot`. Pôvodný príkaz
`scripts/run_uncertainty.py` naďalej funguje ako kompatibilný vstup.

Plný tréning zatiaľ treba vykonať pred porovnaním kvality s MC dropoutom.
Následný cyklus aktívneho učenia zostáva samostatným krokom. Pri ňom sa nesmú
použiť checkpointy naučené na celom tréningovom poole ako úvodné modely.
