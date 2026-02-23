#!/usr/bin/env python3
"""
Script to parse FantaSanremo historical data from Wikipedia and update the JSON file.
Uses web reader MCP tool to fetch data.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import subprocess


def fetch_wikipedia_via_mcp() -> str:
    """Fetch the FantaSanremo Wikipedia page using MCP web reader."""
    # This function will be called with the MCP tool from outside
    # For now, we'll use the data fetched separately
    pass


def parse_editions_from_markdown(markdown_content: str) -> List[Dict[str, Any]]:
    """Parse the editions table from Wikipedia markdown content."""

    editions = []

    # Find the editions table in markdown
    # The table starts with | Edizione | Anno |
    lines = markdown_content.split('\n')

    in_editions_table = False
    header_found = False

    for i, line in enumerate(lines):
        if '| Edizione |' in line and '| Anno |' in line:
            in_editions_table = True
            header_found = True
            continue

        if in_editions_table and header_found:
            # Skip separator line
            if '| --- |' in line:
                continue

            # Parse table row
            if line.startswith('|') and '|' in line and not line.strip().startswith('|---'):
                parts = [p.strip() for p in line.split('|')[1:-1]]  # Remove empty first and last

                # Expected columns: Edizione, Anno, Utenti unici, Squadre iscritte, Leghe, Artista vincitore, Punteggio, Squadre vincitrici, Punteggio squadre
                if len(parts) >= 8:
                    edizione_text = parts[0]
                    # Extract edition number (e.g., "1ª" -> 1, "-" -> 0)
                    if edizione_text.strip() == '-':
                        edizione = 0
                    else:
                        edizione_match = re.search(r'\d+', edizione_text)
                        edizione = int(edizione_match.group()) if edizione_match else 0

                    anno_text = parts[1].replace('\xa0', '').replace(' ', '')
                    anno = int(anno_text) if anno_text.isdigit() else None

                    # Squadre iscritte is column 3
                    squadre_text = parts[3].replace('\xa0', '').replace(' ', '').replace(' ', '')
                    # Remove non-breaking spaces and other unicode spaces
                    squadre_text = re.sub(r'[\s\u00a0\u2009]', '', squadre_text)
                    squadre = int(squadre_text) if squadre_text.isdigit() else 0

                    # Artista vincitore is column 5
                    vincitore = parts[5]
                    # Remove bold markdown and extra whitespace
                    vincitore = re.sub(r"__|'''|\*\*|\([^)]+\)", '', vincitore).strip()

                    # Punteggio is column 6
                    punteggio_text = parts[6].replace('\xa0', '').replace(' ', '')
                    punteggio = int(punteggio_text) if punteggio_text.isdigit() else 0

                    if anno and anno >= 2020:  # Only add FantaSanremo years
                        editions.append({
                            "edizione": edizione,
                            "anno": anno,
                            "squadre": squadre,
                            "vincitore": vincitore,
                            "punteggio": punteggio
                        })

            # Exit table when we hit an empty line or new section
            if not line.strip() and in_editions_table:
                break

    return editions


def parse_artists_from_markdown(markdown_content: str) -> List[Dict[str, Any]]:
    """Parse the artists historical participations table from Wikipedia markdown."""

    artists = []

    lines = markdown_content.split('\n')

    in_artists_table = False
    header_found = False
    year_columns = []

    for i, line in enumerate(lines):
        if '| Cantante |' in line and '| Partecipazioni |' in line:
            in_artists_table = True
            header_found = True
            # Extract year columns from header
            parts = [p.strip() for p in line.split('|')[1:-1]]
            for j, part in enumerate(parts):
                if part in ['2021', '2022', '2023', '2024', '2025']:
                    year_columns.append((j - 1, part))  # Adjust for column index
            continue

        if in_artists_table and header_found:
            # Skip separator line
            if '| --- |' in line:
                continue

            # Parse table row
            if line.startswith('|') and '|' in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]

                if len(parts) >= 6:
                    nome = parts[0]

                    partecipazioni_text = parts[1]
                    partecipazioni = int(partecipazioni_text) if partecipazioni_text.isdigit() else 0

                    # Parse yearly positions
                    yearly_data = {}
                    years = [2021, 2022, 2023, 2024, 2025]

                    for idx, year in enumerate(years):
                        col_idx = idx + 2  # Columns 2-6 correspond to years 2021-2025
                        if col_idx < len(parts):
                            pos_text = parts[col_idx].strip()
                            if pos_text.isdigit():
                                yearly_data[str(year)] = int(pos_text)
                            elif pos_text.lower() in ['non partecipato', 'np', '']:
                                yearly_data[str(year)] = None
                            else:
                                # Try to extract a number
                                pos_match = re.search(r'\d+', pos_text)
                                if pos_match:
                                    yearly_data[str(year)] = int(pos_match.group())
                                else:
                                    yearly_data[str(year)] = None
                        else:
                            yearly_data[str(year)] = None

                    # Check if debuttante
                    debuttante = partecipazioni == 0

                    artists.append({
                        "nome": nome,
                        "partecipazioni": partecipazioni,
                        **yearly_data,
                        "debuttante": debuttante
                    })

            # Exit table when we hit a new section
            if line.strip() and not line.startswith('|') and in_artists_table:
                break

    return artists


def update_json_file(
    editions: List[Dict[str, Any]],
    json_path: Path
) -> None:
    """Update the JSON file with new data while preserving existing structure."""

    # Load existing data
    with open(json_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    # Update albo_d'oro with editions data
    existing_data['albo_oro'] = editions

    # Save updated data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"Updated {json_path}")
    print(f"  - Updated albo_oro with {len(editions)} editions")


def main():
    """Main function to parse Wikipedia data and update JSON."""

    # Wikipedia markdown content (fetched via web reader)
    # This is the content from https://it.wikipedia.org/wiki/FantaSanremo
    wikipedia_markdown = """| FantaSanremo | |
| --- | --- |
| Tipo | Fantasy game |
| Luogo origine | Porto Sant'Elpidio |
| Data origine | 2020-in corso |
| Varianti | - FantaEurovision |
| Regole | |
| N° giocatori | Illimitato |

Il __FantaSanremo__ è un fantasy game basato sul Festival della canzone italiana, consiste nell'organizzare e gestire squadre virtuali formate da cinque degli artisti in gara, di cui uno deve necessariamente essere scelto come capitano[1]. È divenuto un fenomeno nazionale con l'edizione 2022[2], in cui ha raggiunto il mezzo milione di squadre iscritte e molti degli artisti in gara si sono prestati al gioco[3], arrivando persino a coinvolgere il conduttore e direttore artistico Amadeus[4].

Il _FantaSanremo_ nasce nel 2020 dall'idea di un gruppo di amici addetti ai lavori del mondo dello spettacolo (musicisti, insegnanti di musica, tecnici del suono) appassionati del Festival di Sanremo[5], ispiratisi a Fanta Game of Thrones, fantasy game basato sull'omonima serie televisiva. Viene stilato il primo regolamento e coniata la moneta virtuale con cui \\"acquistare\\" gli artisti della propria squadra, chiamata \\"_Baudo_\\" in onore del celebre Pippo, icona del Festival con all'attivo il maggior numero di conduzioni.

Alla fine il numero di iscritti è di 47, la quasi totalità dei quali si ritrova al \\"Bar Corva da Papalina\\" di Porto Sant'Elpidio, in provincia di Fermo, per assistere assieme alle serate della _kermesse._ Nonostante l'esiguo numero di squadre e la dimensione \\"di quartiere\\", il gioco riesce ad attirare l'attenzione di Rai Radio 2, che all'interno del programma Caterpillar intervista telefonicamente uno dei creatori del gioco prima di ogni serata del Festival[6].

Nel 2021, complice la pandemia di COVID-19 e le relative misure restrittive, il FantaSanremo si trasferisce su internet[7], con il motto \\"_Un Team, cinque artisti, un capitano!_\\". All'inizio il sito era stato approntato per ospitare un centinaio di squadre, ma l'interesse da parte di figure molto seguite sui _social_, nonché della lettura del regolamento durante una diretta Instagram da parte di Fedez, danno molta visibilità al gioco, che in pochi giorni vede le squadre iscritte arrivare a 46 962. Visto il repentino successo online, vengono create artisticamente le banconote dei \\"Baudi\\", con Pippo sul taglio da 100 e altre storiche figure della rassegna sugli altri: il maestro Beppe Vessicchio sui 50, Mike Bongiorno sui 20 e Raffaella Carrà sui 10[8].

Nel frattempo comincia ad arrivare anche una prima attenzione mediatica[9], e l'Amministrazione Comunale di Porto Sant'Elpidio concede al team del FantaSanremo l'utilizzo del cittadino Teatro delle Api (ribattezzato \\"Apiston\\") per seguire le dirette del Festival. Durante la settimana sanremese vengono inoltre organizzate dirette sul profilo Instagram del gioco, con ospiti del calibro di Neri Marcorè (direttore artistico del Teatro delle Api) e Paolo Camilli[10], entrambi portoelpidiensi.

Lo Stato Sociale[11], Random[12] e Colapesce Dimartino[13] pronunciano la parola \\"FantaSanremo\\" sul palco dell'Ariston, mentre tutti gli artisti in gara tranne Max Gazzè lo fanno sui canali digitali, più o meno consapevoli di cosa voglia dire.

La classifica artisti è vinta dai Måneskin con 315 punti, davanti a Lo Stato Sociale (300) e Colapesce Dimartino (285)[14], mentre sono tre i \\"fantallenatori\\" a vincere la \\"Gloria Eterna\\" messa in palio[15].

La prima versione del regolamento dell'edizione 2022 è stata pubblicata sul sito del _FantaSanremo_ il giorno di Natale dell'anno precedente[16], con le quotazioni degli artisti rese pubbliche pochi giorni dopo. Le iscrizioni sono state ufficialmente aperte il 31 dicembre[17], annunciate con un video corredato dall'iconica voce di Bruno Pizzul. Dopo tre settimane, le squadre iscritte avevano raggiunto quota 100 000[18], per arrivare a poco più di mezzo milione al termine delle iscrizioni[19]. Radio Italia, che ha pubblicato giornalmente sui propri social aggiornamenti e statistiche relative al gioco, oltre ad articoli dedicati sul proprio sito ufficiale, è stata la radio ufficiale del _FantaSanremo_ 2022[20]. Questo vero e proprio _boom_ ha ovviamente attirato l'attenzione di media e addetti ai lavori[21], fino ad arrivare a Pippo Baudo in persona[22].

Tramite i social, moltissimi fan hanno contattato direttamente gli artisti in gara, portando di fatto alla partecipazione attiva degli artisti[23]: a rompere gli indugi la prima sera è Gianni Morandi, che a fine esibizione urla \\"_FantaSanremo!_\\"", seguito da Michele Bravi, che cita anche \\"Papalina\\", il soprannome del gestore dell'omonimo bar dove il gioco è nato. La seconda serata Sangiovanni prende Amadeus da parte e i due dicono insieme \\"FantaSanremo\\" e \\"Papalina\\". Con il gioco sdoganato sul palco dell'Ariston, inoltre, l'attenzione mediatica è aumentata fino ad arrivare alle conferenze stampa stesse del Festival[24].

Dopo le cinque serate, il podio è risultato composto da Emma (525 punti), Dargen D'Amico (395) e Tananai (365)[25]. Emma, Highsnob, Hu e Tananai sono stati ospiti della diretta finale sulla pagina Instagram del FantaSanremo, in cui hanno conosciuto in diretta la propria posizione in classifica[26].

Il giorno dopo la finale, domenica 6 febbraio, Nicolò \\"Papalina\\" Peroni e Giacomo Piccinini sono stati ospiti di Mara Venier a Domenica in dopo il tormentone \\"Un saluto a zia Mara\\"[27]. L'impatto del gioco sulla cultura popolare italiana viene confermato dall'Istituto Treccani, che inserisce la parola \\"FantaSanremo\\" tra i neologismi del 2022[28].

Per l'edizione 2023, le iscrizioni al _FantaSanremo_ partono il 26 dicembre 2022. Anche in questo caso l'esplosione del fenomeno è evidente: in appena 10 giorni le squadre raggiungono e superano quota 500 000, il numero delle squadre iscritte l'anno precedente. Il 21 gennaio, a meno di un mese dall'apertura delle iscrizioni, il numero di squadre iscritte sfonda il milione, raddoppiando il bilancio del 2022, mentre alla fine il totale sarà di 4 212 694 squadre. A vincere la fantacompetizione è stato il vincitore del Festival, Marco Mengoni, con 670 punti, davanti a Sethu (500) e Rosa Chemical (460).

L'edizione 2024 è partita il 27 dicembre 2023 con l'apertura delle iscrizioni. A vincere la fantacompetizione, che anche quest'anno ha superato le 4 200 000 squadre, sono stati i La Sad, con 486 punti. Seguono Dargen D'Amico e la vincitrice del Festival Angelina Mango, rispettivamente con 460 e 420 punti.

L'edizione 2025 è partita, come l'anno precedente, il 27 dicembre con l'apertura delle iscrizioni. In questa edizione viene leggermente modificato il regolamento, con l'estensione da 5 a 7 cantanti per squadra di cui 5 fanno da titolari e 2 da riserve. Il _FantaSanremo_ 2025, con un totale di più di 5 milioni di squadre iscritte, è stato vinto da Olly (terzo artista ad aggiudicarsi sia il Festival che la fanta competizione dopo i Maneskin e Marco Mengoni) con 475 punti; al secondo posto, come al Festival, Lucio Corsi (440 punti). A completare il podio Sarah Toscano, che ha totalizzato 437 punti.

Nel corso delle edizioni, il regolamento ha subito delle modifiche per adattarsi di volta in volta alla nuova dimensione del gioco, sebbene fossero possibili solo fino alla chiusura delle iscrizioni e non durante la competizione.

Il regolamento del FantaSanremo è per certi aspetti simile a quello del fantacalcio, consistente nel creare una squadra virtuale con personaggi reali. In questo caso si tratta di schierare una squadra composta da 5 artisti big partecipanti al Festival di Sanremo (7 a partire dall'edizione 2025, di cui 5 _titolari_ e 2 _riserve_) uno dei quali deve essere nominato _capitano_ della squadra. Ogni giocatore ha a disposizione 100 _baudi_, la valuta ufficiale di gioco, per acquistare gli artisti in gara. Ogni artista ha un valore, espresso in baudi, che unito a quello dei colleghi non può andare oltre i 100 baudi. Ogni giocatore può iscrivere la sua squadra in una _lega_, cioè un insieme di squadre riferite a un singolo iscritto e che poi esprimerà una propria classifica. Ogni iscritto può creare al massimo cinque squadre e iscriverle in massimo 25 leghe.

Il regolamento, inoltre, prevede diversi _bonus_ e _malus_ assegnati a ogni artista in base alla sua performance sul palco del Teatro Ariston, e che andranno a influire sul punteggio che l'artista stesso totalizzerà al termine del Festival.

Al termine della kermesse viene stilata una classifica sia degli artisti in gara, sia delle squadre sia delle leghe. Vincono la "_Gloria eterna_" l'artista e la squadra che hanno totalizzato il maggior numero di punti.

Fuori dal regolamento invece fa parte il gioco social del _TotoFestival_, dove nella settimana dell'annuncio dei big in gara si può pronosticare gratuitamente, ai fini di statistica, quali artisti parteciperanno al prossimo Festival (dunque sono escluse le _nuove proposte_). La \\"schedina\\" è composta da cinque cantanti scelti da una lista, più altri in una sezione apposita ma che non compaiono nella lista stessa.

| Edizione | Anno | Utenti unici | Squadre iscritte | Leghe | Artista vincitore (numero vittorie) | Punteggio totale artista vincitore | Squadre vincitrici | Punteggio totale squadre vincitrici |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | 2020 | 47 | 47 | 1 | __Piero Pelù__ (1) | 220 | 1 | 655 |
| 1ª | 2021 | 46 962 | 46 962 | 2 | __Måneskin__ (1) | 315 | 3 | 1 325 |
| 2ª | 2022 | 500 012 | 500 012 | 24 042 | __Emma__ (1) | 525 | 19 | 2 200 |
| 3ª | 2023 | 1 991 330 | 4 212 694 | 277 854 | __Marco Mengoni__ (1) | 670 | 14 | 2 696 |
| 4ª | 2024 | 2 870 592 | 4 201 022 | 491 994 | __La Sad__ (1) | 486 | 30 | 2 395 |
| 5ª | 2025 | ? | 5 096 123 | 744 000 | __Olly__ (1) | 475 | 1 | 2 447 |
"""

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    json_path = project_root / 'data' / 'storico_fantasanremo.json'

    print("Parsing editions data from Wikipedia...")
    editions = parse_editions_from_markdown(wikipedia_markdown)
    print(f"  Found {len(editions)} editions")

    print(f"Updating JSON file: {json_path}")
    update_json_file(editions, json_path)

    print("\nSummary of editions found:")
    for ed in editions:
        print(f"  {ed['anno']}: {ed['vincitore']} ({ed['punteggio']} pts, {ed['squadre']:,} teams)")

    print("\nDone!")


if __name__ == '__main__':
    main()
