# Publie ce dossier dans un nouveau depot GitHub public.
# Prerequis : GitHub CLI installe (winget install GitHub.cli) puis "gh auth login".
param([string]$Nom = "jarvis-vision")

git init -b main
git add .
git commit -m "Jarvis Vision v2 : vote OCR, backend ONNX, anonymisation"
gh repo create $Nom --public --source . --remote origin --push
gh repo view $Nom --web
