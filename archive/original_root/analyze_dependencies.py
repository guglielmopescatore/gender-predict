#!/usr/bin/env python3
"""
Script per analizzare le dipendenze tra i file del progetto
per pianificare il refactoring in modo sicuro.
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from pathlib import Path

class DependencyAnalyzer:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.files = {}
        self.imports = {}
        self.class_definitions = {}
        self.function_definitions = {}
        
    def analyze_file(self, file_path: Path) -> Dict:
        """Analizza un singolo file Python."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            imports = []
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            # Cerca anche import tramite regex per catturare import dinamici
            import_pattern = r'from\s+(\w+)\s+import\s+([^#\n]+)|import\s+([^#\n]+)'
            regex_imports = re.findall(import_pattern, content)
            
            for match in regex_imports:
                if match[0] and match[1]:  # from ... import
                    module = match[0]
                    items = [item.strip() for item in match[1].split(',')]
                    for item in items:
                        imports.append(f"{module}.{item}")
                elif match[2]:  # import
                    items = [item.strip() for item in match[2].split(',')]
                    imports.extend(items)
            
            return {
                'path': str(file_path),
                'imports': list(set(imports)),  # Remove duplicates
                'classes': classes,
                'functions': functions,
                'size': file_path.stat().st_size
            }
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {
                'path': str(file_path),
                'imports': [],
                'classes': [],
                'functions': [],
                'size': 0,
                'error': str(e)
            }
    
    def analyze_project(self):
        """Analizza tutti i file Python nel progetto."""
        python_files = list(self.root_dir.glob("*.py"))
        python_files.extend(list(self.root_dir.glob("experiments_improved/*.py")))
        
        for file_path in python_files:
            if file_path.name.startswith('.'):
                continue
                
            analysis = self.analyze_file(file_path)
            self.files[file_path.name] = analysis
            
    def find_internal_dependencies(self) -> Dict[str, List[str]]:
        """Trova le dipendenze interne tra i file del progetto."""
        dependencies = {}
        
        # Lista dei moduli interni (senza estensione .py)
        internal_modules = {f[:-3] for f in self.files.keys() if f.endswith('.py')}
        
        for filename, analysis in self.files.items():
            deps = []
            for imp in analysis['imports']:
                # Pulisci l'import
                module_name = imp.split('.')[0]
                if module_name in internal_modules:
                    deps.append(module_name)
            
            dependencies[filename] = list(set(deps))
            
        return dependencies
    
    def find_shared_classes_functions(self) -> Dict[str, List[str]]:
        """Trova classi e funzioni condivise tra più file."""
        all_classes = {}
        all_functions = {}
        
        for filename, analysis in self.files.items():
            for cls in analysis['classes']:
                if cls not in all_classes:
                    all_classes[cls] = []
                all_classes[cls].append(filename)
                
            for func in analysis['functions']:
                if func not in all_functions:
                    all_functions[func] = []
                all_functions[func].append(filename)
        
        # Trova duplicati
        shared_classes = {k: v for k, v in all_classes.items() if len(v) > 1}
        shared_functions = {k: v for k, v in all_functions.items() if len(v) > 1}
        
        return {
            'classes': shared_classes,
            'functions': shared_functions,
            'all_classes': all_classes,
            'all_functions': all_functions
        }
    
    def generate_migration_plan(self) -> Dict:
        """Genera un piano di migrazione basato sulle dipendenze."""
        dependencies = self.find_internal_dependencies()
        shared = self.find_shared_classes_functions()
        
        # Calcola il "peso" di ogni file basato su dipendenze
        dependency_count = {}
        for filename, deps in dependencies.items():
            dependency_count[filename] = len(deps)
        
        # File senza dipendenze (migrare per primi)
        no_deps = [f for f, deps in dependencies.items() if len(deps) == 0]
        
        # File con molte dipendenze (migrare per ultimi)
        high_deps = sorted(dependency_count.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'dependencies': dependencies,
            'shared_definitions': shared,
            'migration_order': {
                'phase_1_independent': no_deps,
                'phase_2_low_deps': [f for f, c in high_deps if 1 <= c <= 2],
                'phase_3_medium_deps': [f for f, c in high_deps if 3 <= c <= 5],
                'phase_4_high_deps': [f for f, c in high_deps if c > 5],
            }
        }
    
    def print_analysis(self):
        """Stampa l'analisi completa."""
        print("🔍 ANALISI DIPENDENZE PROGETTO")
        print("=" * 50)
        
        # File overview
        print(f"\n📁 File Python trovati: {len(self.files)}")
        for filename, analysis in sorted(self.files.items()):
            size_kb = analysis['size'] / 1024
            print(f"  {filename:<35} {size_kb:>6.1f}KB | "
                  f"Classes: {len(analysis['classes']):>2} | "
                  f"Functions: {len(analysis['functions']):>2} | "
                  f"Imports: {len(analysis['imports']):>2}")
        
        # Dependencies
        dependencies = self.find_internal_dependencies()
        print(f"\n🔗 DIPENDENZE INTERNE:")
        for filename, deps in sorted(dependencies.items()):
            if deps:
                print(f"  {filename:<35} → {', '.join(deps)}")
            else:
                print(f"  {filename:<35} → (nessuna dipendenza)")
        
        # Shared definitions
        shared = self.find_shared_classes_functions()
        if shared['classes']:
            print(f"\n⚠️  CLASSI DUPLICATE:")
            for cls, files in shared['classes'].items():
                print(f"  {cls:<30} → {', '.join(files)}")
        
        if shared['functions']:
            print(f"\n⚠️  FUNZIONI DUPLICATE:")
            for func, files in shared['functions'].items():
                print(f"  {func:<30} → {', '.join(files)}")
        
        # Migration plan
        plan = self.generate_migration_plan()
        print(f"\n🚀 PIANO DI MIGRAZIONE:")
        
        for phase, files in plan['migration_order'].items():
            if files:
                phase_name = phase.replace('phase_', 'Fase ').replace('_', ' - ')
                print(f"\n  {phase_name}:")
                for f in files:
                    deps = len(plan['dependencies'].get(f, []))
                    print(f"    • {f} (dipendenze: {deps})")
        
        print(f"\n💡 RACCOMANDAZIONI:")
        print(f"  1. Inizia con i file senza dipendenze interne")
        print(f"  2. Risolvi le classi duplicate prima della migrazione")
        print(f"  3. Migra utils.py e losses.py per primi (probabilmente usati ovunque)")
        print(f"  4. Migra experiment_manager.py per ultimo (dipende da molti altri)")

def main():
    analyzer = DependencyAnalyzer()
    analyzer.analyze_project()
    analyzer.print_analysis()
    
    # Salva l'analisi in un file
    import json
    plan = analyzer.generate_migration_plan()
    
    with open('migration_analysis.json', 'w') as f:
        # Converti per JSON (rimuovi oggetti non serializzabili)
        json_plan = {
            'files': {k: {
                'classes': v['classes'],
                'functions': v['functions'],
                'imports': v['imports'],
                'size': v['size']
            } for k, v in analyzer.files.items()},
            'dependencies': plan['dependencies'],
            'migration_order': plan['migration_order']
        }
        json.dump(json_plan, f, indent=2)
    
    print(f"\n📄 Analisi salvata in migration_analysis.json")

if __name__ == "__main__":
    main()