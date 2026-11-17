"""
Generates clean, modern SVG visual figures for Lab 3 results.
"""

import os

os.makedirs("images", exist_ok=True)

# 1. Models Comparison Chart (Bar Chart)
svg_comparison = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 480" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <!-- Title -->
  <text x="425" y="40" text-anchor="middle" font-size="22" font-weight="bold" fill="#1e293b">Lab 3 - Part 1: Sequence Models Performance Comparison</text>
  <text x="425" y="65" text-anchor="middle" font-size="14" fill="#64748b">Evaluation on Arabic NLP Classification Task (Dataset: AI &amp; Deep Learning)</text>
  
  <!-- Legend -->
  <rect x="250" y="85" width="18" height="14" rx="3" fill="#3b82f6"/>
  <text x="275" y="97" font-size="13" fill="#334155" font-weight="600">Accuracy</text>
  
  <rect x="370" y="85" width="18" height="14" rx="3" fill="#10b981"/>
  <text x="395" y="97" font-size="13" fill="#334155" font-weight="600">F1-Score (Weighted)</text>
  
  <rect x="545" y="85" width="18" height="14" rx="3" fill="#8b5cf6"/>
  <text x="570" y="97" font-size="13" fill="#334155" font-weight="600">BLEU Score</text>

  <!-- Grid lines -->
  <line x1="100" y1="380" x2="780" y2="380" stroke="#cbd5e1" stroke-width="1.5"/>
  <line x1="100" y1="315" x2="780" y2="315" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="85" y="320" font-size="12" fill="#94a3b8" text-anchor="end">0.20</text>
  
  <line x1="100" y1="250" x2="780" y2="250" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="85" y="255" font-size="12" fill="#94a3b8" text-anchor="end">0.40</text>
  
  <line x1="100" y1="185" x2="780" y2="185" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="85" y="190" font-size="12" fill="#94a3b8" text-anchor="end">0.60</text>
  
  <line x1="100" y1="120" x2="780" y2="120" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="85" y="125" font-size="12" fill="#94a3b8" text-anchor="end">0.80</text>

  <!-- Group 1: Simple RNN (Acc: 0.5122, F1: 0.5332, BLEU: 0.4352) -->
  <!-- Base Y = 380. Height = Val * 325 -->
  <!-- Acc = 166.4 -> Y = 213.6 -->
  <!-- F1 = 173.3 -> Y = 206.7 -->
  <!-- BLEU = 141.4 -> Y = 238.6 -->
  <g transform="translate(140, 0)">
    <rect x="0" y="213.6" width="35" height="166.4" rx="4" fill="#3b82f6"/>
    <text x="17.5" y="205" font-size="11" font-weight="bold" fill="#1e40af" text-anchor="middle">0.512</text>
    
    <rect x="42" y="206.7" width="35" height="173.3" rx="4" fill="#10b981"/>
    <text x="59.5" y="198" font-size="11" font-weight="bold" fill="#065f46" text-anchor="middle">0.533</text>
    
    <rect x="84" y="238.6" width="35" height="141.4" rx="4" fill="#8b5cf6"/>
    <text x="101.5" y="230" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">0.435</text>
    
    <text x="59.5" y="405" font-size="14" font-weight="600" fill="#1e293b" text-anchor="middle">Simple RNN</text>
  </g>

  <!-- Group 2: Bidirectional RNN (Acc: 0.4268, F1: 0.3939, BLEU: 0.4292) -->
  <!-- Acc = 138.7 -> Y = 241.3 -->
  <!-- F1 = 128.0 -> Y = 252.0 -->
  <!-- BLEU = 139.5 -> Y = 240.5 -->
  <g transform="translate(305, 0)">
    <rect x="0" y="241.3" width="35" height="138.7" rx="4" fill="#3b82f6"/>
    <text x="17.5" y="233" font-size="11" font-weight="bold" fill="#1e40af" text-anchor="middle">0.427</text>
    
    <rect x="42" y="252.0" width="35" height="128.0" rx="4" fill="#10b981"/>
    <text x="59.5" y="244" font-size="11" font-weight="bold" fill="#065f46" text-anchor="middle">0.394</text>
    
    <rect x="84" y="240.5" width="35" height="139.5" rx="4" fill="#8b5cf6"/>
    <text x="101.5" y="232" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">0.429</text>
    
    <text x="59.5" y="405" font-size="14" font-weight="600" fill="#1e293b" text-anchor="middle">Bidirectional RNN</text>
  </g>

  <!-- Group 3: LSTM (Acc: 0.4512, F1: 0.4262, BLEU: 0.4309) -->
  <!-- Acc = 146.6 -> Y = 233.4 -->
  <!-- F1 = 138.5 -> Y = 241.5 -->
  <!-- BLEU = 140.0 -> Y = 240.0 -->
  <g transform="translate(470, 0)">
    <rect x="0" y="233.4" width="35" height="146.6" rx="4" fill="#3b82f6"/>
    <text x="17.5" y="225" font-size="11" font-weight="bold" fill="#1e40af" text-anchor="middle">0.451</text>
    
    <rect x="42" y="241.5" width="35" height="138.5" rx="4" fill="#10b981"/>
    <text x="59.5" y="233" font-size="11" font-weight="bold" fill="#065f46" text-anchor="middle">0.426</text>
    
    <rect x="84" y="240.0" width="35" height="140.0" rx="4" fill="#8b5cf6"/>
    <text x="101.5" y="232" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">0.431</text>
    
    <text x="59.5" y="405" font-size="14" font-weight="600" fill="#1e293b" text-anchor="middle">LSTM</text>
  </g>

  <!-- Group 4: GRU (Acc: 0.4024, F1: 0.4001, BLEU: 0.4275) -->
  <!-- Acc = 130.8 -> Y = 249.2 -->
  <!-- F1 = 130.0 -> Y = 250.0 -->
  <!-- BLEU = 138.9 -> Y = 241.1 -->
  <g transform="translate(635, 0)">
    <rect x="0" y="249.2" width="35" height="130.8" rx="4" fill="#3b82f6"/>
    <text x="17.5" y="241" font-size="11" font-weight="bold" fill="#1e40af" text-anchor="middle">0.402</text>
    
    <rect x="42" y="250.0" width="35" height="130.0" rx="4" fill="#10b981"/>
    <text x="59.5" y="242" font-size="11" font-weight="bold" fill="#065f46" text-anchor="middle">0.400</text>
    
    <rect x="84" y="241.1" width="35" height="138.9" rx="4" fill="#8b5cf6"/>
    <text x="101.5" y="233" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">0.428</text>
    
    <text x="59.5" y="405" font-size="14" font-weight="600" fill="#1e293b" text-anchor="middle">GRU</text>
  </g>

  <!-- Bottom subtitle -->
  <text x="425" y="450" text-anchor="middle" font-size="12" fill="#94a3b8">Étudiant : Yahya AHMANE - LSI 3ème Année - FST Tanger</text>
</svg>"""

with open("images/models_comparison.svg", "w", encoding="utf-8") as f:
    f.write(svg_comparison)

# 2. GPT-2 Training Loss Curve
svg_loss = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 400" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <text x="375" y="38" text-anchor="middle" font-size="20" font-weight="bold" fill="#1e293b">Lab 3 - Part 2: GPT-2 Fine-Tuning Convergence</text>
  <text x="375" y="62" text-anchor="middle" font-size="13" fill="#64748b">Training Loss over 3 Epochs on Custom Deep Learning Dataset</text>

  <!-- Grid lines -->
  <line x1="120" y1="320" x2="650" y2="320" stroke="#cbd5e1" stroke-width="1.5"/>
  <line x1="120" y1="260" x2="650" y2="260" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="105" y="265" font-size="12" fill="#94a3b8" text-anchor="end">6.00</text>
  
  <line x1="120" y1="200" x2="650" y2="200" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="105" y="205" font-size="12" fill="#94a3b8" text-anchor="end">6.50</text>
  
  <line x1="120" y1="140" x2="650" y2="140" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="105" y="145" font-size="12" fill="#94a3b8" text-anchor="end">7.00</text>
  
  <line x1="120" y1="80" x2="650" y2="80" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4"/>
  <text x="105" y="85" font-size="12" fill="#94a3b8" text-anchor="end">7.50</text>

  <!-- X axis labels -->
  <!-- Epoch 1: X = 200, Y for 7.4175 -> 80 + (7.5-7.4175)/0.5 * 60 = 89.9 -->
  <!-- Epoch 2: X = 385, Y for 6.7380 -> 140 + (7.0-6.7380)/0.5 * 60 = 171.4 -->
  <!-- Epoch 3: X = 570, Y for 6.1652 -> 200 + (6.5-6.1652)/0.5 * 60 = 240.2 -->
  
  <!-- Shaded area under curve -->
  <polygon points="200,89.9 385,171.4 570,240.2 570,320 200,320" fill="#3b82f6" fill-opacity="0.1"/>
  
  <!-- Loss line -->
  <polyline points="200,89.9 385,171.4 570,240.2" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Data points -->
  <circle cx="200" cy="89.9" r="6" fill="#1d4ed8" stroke="#ffffff" stroke-width="2"/>
  <text x="200" y="75" font-size="12" font-weight="bold" fill="#1d4ed8" text-anchor="middle">7.4175</text>
  <text x="200" y="342" font-size="13" font-weight="600" fill="#334155" text-anchor="middle">Epoch 1</text>

  <circle cx="385" cy="171.4" r="6" fill="#1d4ed8" stroke="#ffffff" stroke-width="2"/>
  <text x="385" y="157" font-size="12" font-weight="bold" fill="#1d4ed8" text-anchor="middle">6.7380</text>
  <text x="385" y="342" font-size="13" font-weight="600" fill="#334155" text-anchor="middle">Epoch 2</text>

  <circle cx="570" cy="240.2" r="6" fill="#10b981" stroke="#ffffff" stroke-width="2"/>
  <text x="570" y="226" font-size="12" font-weight="bold" fill="#047857" text-anchor="middle">6.1652</text>
  <text x="570" y="342" font-size="13" font-weight="600" fill="#334155" text-anchor="middle">Epoch 3</text>
  
  <text x="375" y="380" text-anchor="middle" font-size="12" fill="#94a3b8">Convergence régulière observée avec l'optimiseur AdamW (lr=5e-5)</text>
</svg>"""

with open("images/gpt2_training_loss.svg", "w", encoding="utf-8") as f:
    f.write(svg_loss)

# 3. NLP Pipeline Architecture Diagram
svg_pipeline = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 220" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <text x="460" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">Architecture du Pipeline de Traitement NLP (Langue Arabe)</text>
  
  <!-- Step 1 -->
  <g transform="translate(30, 60)">
    <rect width="140" height="90" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="2"/>
    <text x="70" y="35" text-anchor="middle" font-size="13" font-weight="bold" fill="#1d4ed8">1. Collecte</text>
    <text x="70" y="55" text-anchor="middle" font-size="11" fill="#475569">Scraping Web</text>
    <text x="70" y="70" text-anchor="middle" font-size="10" fill="#64748b">BeautifulSoup</text>
  </g>
  <path d="M 175 105 L 205 105" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Step 2 -->
  <g transform="translate(210, 60)">
    <rect width="145" height="90" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="2"/>
    <text x="72.5" y="35" text-anchor="middle" font-size="13" font-weight="bold" fill="#15803d">2. Nettoyage</text>
    <text x="72.5" y="55" text-anchor="middle" font-size="11" fill="#475569">Tashkeel &amp; Tatweel</text>
    <text x="72.5" y="70" text-anchor="middle" font-size="10" fill="#64748b">Normalisation Alif/Yaa</text>
  </g>
  <path d="M 360 105 L 390 105" stroke="#94a3b8" stroke-width="2"/>

  <!-- Step 3 -->
  <g transform="translate(395, 60)">
    <rect width="145" height="90" rx="8" fill="#faf5ff" stroke="#a855f7" stroke-width="2"/>
    <text x="72.5" y="35" text-anchor="middle" font-size="13" font-weight="bold" fill="#7e22ce">3. Morphologie</text>
    <text x="72.5" y="55" text-anchor="middle" font-size="11" fill="#475569">Stopwords NLTK</text>
    <text x="72.5" y="70" text-anchor="middle" font-size="10" fill="#64748b">ISRIStemmer</text>
  </g>
  <path d="M 545 105 L 575 105" stroke="#94a3b8" stroke-width="2"/>

  <!-- Step 4 -->
  <g transform="translate(580, 60)">
    <rect width="145" height="90" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="2"/>
    <text x="72.5" y="35" text-anchor="middle" font-size="13" font-weight="bold" fill="#b45309">4. Discrétisation</text>
    <text x="72.5" y="55" text-anchor="middle" font-size="11" fill="#475569">Score 0-10 -> 3 Classes</text>
    <text x="72.5" y="70" text-anchor="middle" font-size="10" fill="#64748b">Vocab &amp; Padding</text>
  </g>
  <path d="M 730 105 L 760 105" stroke="#94a3b8" stroke-width="2"/>

  <!-- Step 5 -->
  <g transform="translate(765, 60)">
    <rect width="135" height="90" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="2"/>
    <text x="67.5" y="35" text-anchor="middle" font-size="13" font-weight="bold" fill="#b91c1c">5. Modélisation</text>
    <text x="67.5" y="55" text-anchor="middle" font-size="11" fill="#475569">RNN / BiRNN</text>
    <text x="67.5" y="70" text-anchor="middle" font-size="10" fill="#64748b">LSTM / GRU</text>
  </g>

  <text x="460" y="195" text-anchor="middle" font-size="12" fill="#94a3b8">Étudiant : Yahya AHMANE - LSI 3ème Année</text>
</svg>"""

with open("images/part1_architecture.svg", "w", encoding="utf-8") as f:
    f.write(svg_pipeline)

print("Generated visual diagrams successfully in images/ directory:")
print(" - images/models_comparison.svg")
print(" - images/gpt2_training_loss.svg")
print(" - images/part1_architecture.svg")
