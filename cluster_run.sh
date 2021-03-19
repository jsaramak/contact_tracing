
#!/bin/bash
#SBATCH --job-name=contact_tracing
#SBATCH --output=contact_tracing_%A_%a.out
#SBATCH --array=0-100
#SBATCH --time=04:30:00
#SBATCH --mem=1G

python contact_tracing.py
