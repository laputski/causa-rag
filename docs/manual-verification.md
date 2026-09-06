# Checking it by hand

Every claim in this repository is guarded by a test, and a test can only ask
what somebody thought to ask. This is the other check: a person, a screen, and
forty minutes.

It is written as seven steps. Each says what to do, what should be on screen
afterwards, and what it means if something else is. The numbers below were
measured on a machine following these steps, so a difference is worth reading
and never dismissing; where a number legitimately varies, it says so.

There are no screenshots. A verification document is compared against your own
screen, and an exact value can be checked where a picture can only be found to
look about right.

---

## 1. Bring the services up

```bash
make infra
python3 -m tools.doctor
```

**You should see** seven lines, each with a tick: docker, qdrant, opensearch,
mongodb, redis, langfuse, neo4j. The last is optional and sits behind a compose
profile; `– neo4j :7687 profile 'graph' not started (optional)` is a normal
answer and everything below still works.

**If a line is red** the report names the command that fixes it. Nothing after
this step can be trusted while one of the first four is down: qdrant and
opensearch hold the two halves of retrieval, mongodb holds every run.

**A note on this machine's history.** `make infra` names the compose project
after the directory. If you have an older stack running under a different
project name, the two collide on a port and neither comes up cleanly. `docker
ps` tells you which names are running.

## 2. Look at the realm that is meant to work

```bash
make demo
make ui        # in another terminal, if it is not already running
```

Open <http://localhost:5173/overview?realm=demo>.

**You should see** the realm named Demo, a band counting one corpus, fifteen
questions, one prompt, and however many runs you have made. Under it, five
setup steps, all ticked, and a Status column with every service up.

**What it proves.** The platform reads its own installation, and the realm
that ships with it is complete. Nothing here is about failure detection yet:
this is the control for everything after it.

## 3. Look at the realm that is meant to be broken

```bash
make proving-ground
```

Open <http://localhost:5173/overview?realm=proving-ground>.

**You should see** the realm named Proving Ground with an amber mark reading
*broken on purpose* beside the title, a description saying that a red
diagnostic here is the expected outcome, two corpora, and thirty-eight
questions in two sets.

**What it proves.** A realm carrying deliberate defects says so where it is
named. Without that mark its diagnostics read as a broken installation, and the
installation check would be reporting on a realm built to fail.

**If the mark is missing**, the realm was seeded before the mark existed. Seed
it again with `python3 -m tools.seed_proving_ground --force`.

## 4. Read the healthy half

Open <http://localhost:5173/data/health?realm=proving-ground&corpus_id=base-ru>.

**You should see** 220 chunks, 0 duplicates, 0 without a structural path, 0
carrying a heading and no body, and under *What to fix* the line **The corpus
looks healthy**, followed by one informational note about two near-duplicate
chunks.

**About that note.** It is correct and it is not a defect. The corpus holds
twenty-five service cards for one family of instruments, and two of them differ
by a model code and a figure. The check exists to put such a pair in front of a
person to judge, which is what it is doing.

**What matters for the next step** is what is *absent*: there is no finding
about mixed languages. Write that down. Half a check proves nothing, and this
is the half that people skip.

## 5. Put one named defect into the corpus

```bash
python3 -m tools.corpus_mutate --list
```

**You should see** six defects, each with the catalogue entries it is meant to
make observable: flattened headings (F03), duplicated documents (F01, F26), a
dropped number (F04), a repeated number (F04), sections cut to fragments (F06),
and a second language (F14, F24).

Now make a damaged copy and load it beside the healthy one:

```bash
python3 -m tools.corpus_mutate corpus/proving-ground/base-ru \
  --defect add_a_second_language --out /tmp/broken-ru

USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest /tmp/broken-ru \
  --strategy structure_aware --corpus-id base-ru-mixed \
  --language ru_be --realm-id proving-ground
```

**You should see** `40 documents in, 56 out, defect 'add_a_second_language'`
from the first command, and from the second `Ingested 252 chunks` followed by
`Registered as corpus 'base-ru-mixed' of realm 'proving-ground'`.

**The second line matters.** Without it the corpus would exist in the index and
in no list on any screen, and the step after this would have nothing to open.

**If the tool refuses**, read what it says. It refuses when the corpus cannot
carry the defect asked of it, which is deliberate: a mutation that changes the
files and provokes nothing is worse than one that stops.

## 6. Read the damaged half

Open
<http://localhost:5173/data/health?realm=proving-ground&corpus_id=base-ru-mixed>.

**You should see** 252 chunks and, under *What to fix*, **Mixed languages in
the corpus** carrying two links, **F14** and **F24**. The informational note
about near duplicates is still there, now linked to **F26**.

**What it proves, and only this.** The same check, on the same platform, over
two corpora that differ by one named change: silent about languages on one,
speaking on the other. That is the whole of what a paired observation claims,
and it is more than either half claims alone. A check that fires on everything
passes the loud half every time.

**Follow one of the links.** It opens the catalogue at that entry. If the entry
cannot occur in the architecture the page is showing, the page moves to one
where it can, and says which.

## 7. Read what the platform cannot do

Open <http://localhost:5173/atlas?realm=proving-ground>.

**You should see** an architecture chosen by its coordinates, each coordinate
linking to its definition at ragworld.org, and a count of the entries that
apply to that architecture: 36 of 41 for the hybrid point, 28 for the dense
one, 35 for the graph one. Switching the architecture changes the count and the
list, and never shows a total across all three.

**Every entry carries one of four states.** Caught by a named signal. Visible
in the data, for a person to read. Claimed and unproven. Not detected at all.
The last is shown beside the others on purpose: a catalogue showing only what
it catches is advertising.

**Open an entry that is not detected.** It says what is missing: a check nobody
has written, or knowledge the platform does not record. That sentence is the
point of the whole page.

**On the Signals tab**, each named judgement of the platform is listed with the
entries it is evidence for. Three signals are listed as evidence for nothing,
and say why: they report that a check could not be made, which is the absence
of a verdict, and no verdict at all.

---

## What this does not check

**The runs.** Steps 4 to 6 read a corpus, which takes seconds. Staging a
failure that only shows in a run means making a run, which takes minutes and a
generation model. Those pairs are automated instead:

```bash
make test-proving-ground
```

That suite makes a real run against a real index for each pair and reads the
signal off what the run recorded. It needs the whole stack, the real embedding
model, and the loaded indexes, and it skips loudly, never passing, when one
of those is missing. Thirteen catalogue entries are reproduced that way today;
the report says which.

```bash
python3 -m tools.atlas_report --point hybrid
```

**The numbers on the screens.** That every metric is computed correctly is what
the unit suite is for, and it is not something a person can check by looking.
What a person can check is that the platform says which layer lost an answer,
and that the answer it names matches the questions underneath it.

**Anything about your own corpus.** The proving ground is two corpora this
repository writes. That a defect is caught here says the check works; it does
not say your corpus has or has not got that defect. That is what a run against
your own data is for.
