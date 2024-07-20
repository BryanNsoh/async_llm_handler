# Ensure we're in the project root
Set-Location $PSScriptRoot

# Read the current version
$current_version = Get-Content version.txt
Write-Host "Current version: $current_version"

# Prompt for the new version
$new_version = Read-Host -Prompt "Enter new version number"

# Update version.txt
Set-Content -Path version.txt -Value $new_version

# Update README.md if necessary
(Get-Content README.md) -replace "version $current_version", "version $new_version" | Set-Content README.md

# Git operations
git add .
git commit -m "Update to version $new_version"
git tag "v$new_version"
git push origin main --tags

# Build and upload to PyPI
python -m build
twine upload dist/*

Write-Host "Version $new_version has been released!"