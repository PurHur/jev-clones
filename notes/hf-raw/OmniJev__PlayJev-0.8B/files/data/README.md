# PlayJev Training Data

Every decision PlayJev-0.8B was trained on: 2,166,984 records
from 11,416 episodes of ten browser games, with the teacher's target for each one. Code, games and the training
recipe are at [github.com/OmniJev/PlayJev](https://github.com/OmniJev/PlayJev).

The frames are drawn again on your machine. The games run on a virtual clock with a seeded `Math.random`, so an
episode is fixed by its seed and the moves taken, and both are in every record:

```bash
git clone https://github.com/OmniJev/PlayJev && cd PlayJev
pip install -r requirements.txt && playwright install chromium
hf download OmniJev/PlayJev-0.8B --include "data/*" --local-dir .
python -m playjev.rebuild data --pages 16
```

`playjev.rebuild` replays each episode in headless Chromium (about two hours at 16 pages), checks the score and
the clock at every step, writes each frame where its record points and counts the frames whose MD5 matches
`frame_md5`. On x86-64 Linux every frame matches the one we trained on, byte for byte (checked on two clusters). On
ARM Linux the states are the same and five games match byte for byte; Breakout, Floppy Bird, Space Invaders,
Mario and Racer differ by a few grey levels where sprites are scaled. With the frames in place,
`bash scripts/reproduce.sh` keeps every complete shard and trains on it.

## Layout

`<game>/<round>_<shard>/records.jsonl`, one line per decision, three shards per round.

| Round | Played by | Records |
|---|---|---:|
| `clone` | the teachers, 2 to 30 percent random moves | 966,744 |
| `dagger1` | the cloning model, 5 percent random moves | 400,080 |
| `dagger2` | version 1 | 400,080 |
| `dagger3` | the replay model version 3 continues from | 400,080 |

Each game has 100,008 cloning records and 40,008 per DAgger round, except Floppy Bird, whose `clone_b` collector
died at launch in our run: every model was trained without it, so it is absent here too.

## Fields

| Field | Meaning |
|---|---|
| `game`, `shard` | where the record lives |
| `seed`, `episode`, `step` | the episode seed (unique within a shard) and the step within that episode, from 0 |
| `frame`, `prev_frame` | the frame's path inside the shard, and the previous step's (null at step 0) |
| `frame_md5` | MD5 of the 448 px JPEG we trained on |
| `actions` | the game's move names, in option order |
| `teacher_probs`, `teacher_action` | the teacher's soft target over the moves, and its best move |
| `taken_action` | the move actually played, which the rebuild replays |
| `actor` | `teacher` for the cloning round, `local` when the model played |
| `score`, `reward`, `done`, `t` | game score before the move, the score change, episode end, virtual clock in ms |

## Licence

Apache-2.0, like the code and the weights. The frames you rebuild show each game's own art, under the licence of
the game it comes from (see the repository's licence section).
