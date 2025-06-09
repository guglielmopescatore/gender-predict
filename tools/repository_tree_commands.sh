#!/bin/bash
# Comandi per analizzare la struttura completa del repository

echo "🌳 GENDER PREDICTION REPOSITORY ANALYSIS"
echo "========================================"

cd ~/Git_Repositories/gender-predict

echo ""
echo "🔍 ENVIRONMENT CHECK:"
echo "====================="
if [ -d "env/" ]; then
    echo "📁 Python virtual environment found: env/"
    if grep -q "env/" .gitignore 2>/dev/null; then
        echo "✅ env/ properly excluded in .gitignore"
    else
        echo "⚠️  RECOMMENDATION: Add 'env/' to .gitignore"
    fi
    echo "ℹ️  env/ will be excluded from all analysis below"
else
    echo "✅ No env/ directory found"
fi

echo ""
echo "📋 BRANCH STRUCTURE:"
echo "===================="
git branch -a

echo ""
echo "📊 RECENT COMMITS (Main):"
echo "=========================="
git log --oneline -5 main

echo ""
echo "📊 RECENT COMMITS (Development):"
echo "================================="
git log --oneline -5 development

echo ""
echo "📊 RECENT COMMITS (Commercial-Foundation):"
echo "==========================================="
git log --oneline -5 commercial-foundation

echo ""
echo "🌳 REPOSITORY TREE (Main Branch):"
echo "=================================="
# Check for Python virtual environment
if [ -d "env/" ]; then
    echo "📁 Python virtual environment detected: env/ (excluded from analysis)"
    if grep -q "env/" .gitignore 2>/dev/null; then
        echo "✅ env/ correctly excluded in .gitignore"
    else
        echo "⚠️  env/ should be added to .gitignore"
    fi
    echo ""
fi

tree -a -I '.git|env|__pycache__|*.pyc|*.pyo|.env' -L 3

echo ""
echo "📁 DETAILED API DIRECTORY:"
echo "=========================="
ls -la api/

echo ""
echo "📁 DETAILED SRC DIRECTORY:"
echo "=========================="
find src/ -type f -name "*.py" | head -10

echo ""
echo "📁 DETAILED SCRIPTS DIRECTORY:"
echo "==============================="
ls -la scripts/

echo ""
echo "📁 EXPERIMENTS DIRECTORY:"
echo "=========================="
ls -la experiments/ | head -5

echo ""
echo "📁 MODELS DIRECTORY:"
echo "===================="
ls -la models/ | head -5

echo ""
echo "🔧 CONFIGURATION FILES:"
echo "======================="
find . -name "*.py" -path "./api/*" -exec basename {} \;
find . -name "*.md" -maxdepth 1
find . -name "*.txt" -maxdepth 1
find . -name "*.yml" -maxdepth 1 -o -name "*.yaml" -maxdepth 1

echo ""
echo "📊 REPOSITORY STATISTICS:"
echo "========================="
echo "Total Python files: $(find . -name "*.py" -not -path "./env/*" | wc -l)"
echo "Total directories: $(find . -type d -not -path "./env/*" | wc -l)"
echo "Total files (excluding env/): $(find . -type f -not -path "./env/*" | wc -l)"
echo "Repository size (excluding env/): $(du -sh --exclude=env . | cut -f1)"

echo ""
echo "🔍 KEY FILES ANALYSIS:"
echo "======================"
echo "API Files:"
find api/ -name "*.py" 2>/dev/null || echo "No API directory"

echo ""
echo "Configuration Files:"
find . -name "config*.py" -o -name "requirements*.txt" -o -name "setup.py"

echo ""
echo "Documentation Files:"
find . -name "README*" -o -name "*.md" | head -10

echo ""
echo "🚀 DEPLOYMENT FILES:"
echo "===================="
find . -name "*deploy*" -o -name "*modal*" -o -name "Dockerfile*"

# If you want to save the output to a file
echo ""
echo "💾 SAVING TREE TO FILE:"
echo "======================="
tree -a -I '.git|env|__pycache__|*.pyc|*.pyo|.env' > repository_tree.txt
echo "Tree saved to repository_tree.txt (env/ excluded)"

echo ""
echo "📋 FINAL SUMMARY:"
echo "=================="
echo "✅ Repository structure analyzed"
echo "✅ Branch information collected"
echo "✅ Key directories mapped"
echo "✅ Configuration files identified"
echo "✅ Ready for commercial development analysis"
