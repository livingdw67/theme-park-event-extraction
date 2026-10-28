# Annotation guide

The test set labels four entity types in sentences from Wikipedia. Spans are exact character ranges,
and scoring counts a prediction as correct only if its start, end and label all match.

| Label | Tag | Don't tag |
|---|---|---|
| **PARK** | Named theme parks, water parks and resorts: *Universal Studios Florida*, *Islands of Adventure*, *Universal Orlando Resort*, *Magic Kingdom* | Lands and areas inside a park (*The Wizarding World of Harry Potter – Hogsmeade*, *Jurassic Park* the land), companies (*Universal Parks & Resorts*, *Comcast*), hotels |
| **ATTRACTION** | Named rides, roller coasters, shows and walk-through attractions, current or closed: *Revenge of the Mummy*, *Kongfrontation*, *Dueling Dragons* | Films, franchises and characters the ride is based on (*The Mummy*, *King Kong*), restaurants and shops, lands, ride manufacturers and ride models (*Intamin*, *Boomerang*) |
| **EVENT** | Named recurring or seasonal park events, with any edition number: *Halloween Horror Nights 33*, *HHN*, *Mardi Gras*, *Rock the Universe*, *Grinchmas* | Haunted houses and scare zones inside an event, one-off news events (hurricanes, the COVID-19 pandemic), film premieres |
| **DATE** | Calendar expressions: full dates, month and year, seasons with a year, bare years, decades, and their modifiers: *September 8, 2002*, *late 2024*, *summer 2025*, *2002*, *the 1990s* (tag *1990s*) | Durations (*two years*), times of day, ages, counts, relative dates (*last year*, *the following season*) |

Boundaries:
- Include the full official name as written, including a leading "The" only when it is part of the
  name (*The Amazing Adventures of Spider-Man*), and edition numbers on events.
- Leave out trailing punctuation, possessive 's, and descriptive words (*the ride*, *roller coaster*).
- A date range written as one expression (*September 1–3, 2023*) is one span; *from September 1 to
  November 4, 2023* is two spans.
- Where an attraction is named after its park's land or a film and the sentence means the ride, tag
  the ride name (*Jurassic Park River Adventure*); where it means the film or land, don't.
- Modifiers *early*, *mid* and *late* go inside a DATE span.
- A possessive brand prefix is left out (*Universal's* Halloween Horror Nights, *Universal Orlando's*
  Horror Make-Up Show), but a name that officially starts with *Universal* keeps it (*Universal
  Studios Florida*, *Universal Epic Universe*).
- Bare month names with no day or year (*two nights in January*) are not DATEs. A year range written
  as one expression (*1996–2017*) is one span.
- Parks that were planned under a name but never built under it still count as PARK (*Cartoon
  World*). In-ride scenes, fictional places and queue signage are not tagged.
- Where sentence splitting cut an attraction's name in two, the visible part is still tagged.

The draft labels were written by hand with the rule and GLiNER outputs shown alongside as a
checklist, then reviewed sentence by sentence before scoring. Both taggers' outputs were treated
as suggestions, never accepted automatically.
