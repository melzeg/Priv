# Email Labeler

Een Mistral Workflow die dagelijks nieuwe Gmail-threads classificeert en automatisch
de labels **Wijn**, **Muziek** of **Zeilen** toekent.

## Hoe het werkt

1. De workflow zoekt Gmail-threads van de laatste paar dagen die nog geen van de drie
   labels hebben (`newer_than:Nd -label:wijn -label:muziek -label:zeilen ...`).
2. Elke thread wordt met `mistral-small-latest` geclassificeerd in `wijn`, `muziek`,
   `zeilen` of `geen` (gestructureerde output).
3. Bij een match wordt het bijbehorende Gmail-label op de hele thread gezet
   (`label_thread`, met fallback naar `label_message` per bericht).
4. De run retourneert een samenvatting: aantal threads gezien, gelabeld, en per thread
   de categorie.

De labels worden bij de eerste run automatisch aangemaakt als ze nog niet bestaan.

## Input

| Veld          | Default | Betekenis                                        |
| ------------- | ------- | ------------------------------------------------ |
| `dry_run`     | `false` | Classificeer maar label niets                    |
| `days_back`   | `3`     | Hoe ver terug Gmail doorzocht wordt              |
| `max_threads` | `25`    | Maximum aantal threads per run                   |

## Lokaal draaien

```bash
cp .env.example .env   # vul MISTRAL_API_KEY in
uv sync --all-groups
uv run pytest
uv run python -m worker.entrypoints.worker
```

## Deployment

De worker wordt als managed deployment gebouwd vanuit deze repository:

- Dockerfile: `deploy/docker/Dockerfile.worker`
- Entrypoint: `python -m worker.entrypoints.worker`

De workflow draait onbeheerd (dagelijkse schedule) en gebruikt de Gmail-connector met
`run_as="deployment"`: de Gmail-connector moet daarom een **credential op
workspace-/deployment-niveau** hebben in Mistral Studio, naast de persoonlijke
user-credential.
