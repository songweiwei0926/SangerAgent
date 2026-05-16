# SangerAgent

**Automated Sanger sequencing analysis for SNV detection and CRISPR editing quantification.**

SangerAgent processes AB1 chromatogram files to detect single nucleotide variants,
quantify CRISPR editing efficiency, and annotate variants with ClinVar clinical significance.

## Quick Start

```bash
# Clone and start
git clone https://github.com/your-org/SangerAgent.git
cd SangerAgent
docker-compose up

# Open the web interface
open http://localhost:3000

# API documentation
open http://localhost:8000/docs
```

## Features

- **SNV Detection** — Mixed-peak quantification with heterozygosity detection
- **Editing Analysis** — ICE, TIDE, and BEAT for CRISPR efficiency estimation
- **ClinVar Annotation** — Automated clinical significance lookup with caching
- **Batch Processing** — Multi-file upload with aggregate CSV reports
- **REST API** — Full OpenAPI documentation at `/docs`

## Documentation

- **[Complete User Guide](docs/HUMAN_README.md)** — Setup, configuration, API examples, troubleshooting
- **[Methods](docs/METHODS.md)** — Publication-quality description of all algorithms
- **[AI Context](docs/AI_CONTEXT.md)** — Architecture and interface reference for developers

## Requirements

- Docker 24.0+ and Docker Compose 2.20+
- 4 GB RAM (8 GB recommended)
- 10 GB disk space

## Optional: hg38 Genome Index

For fast local alignment (recommended for production):

```bash
bash scripts/download_hg38.sh
```

Without the index, NCBI BLAST is used as a fallback (slower, requires internet).

## Development Setup

```bash
bash scripts/setup.sh
```

## License

MIT License — see [LICENSE](LICENSE) for details.
