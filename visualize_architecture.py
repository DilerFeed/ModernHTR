import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from config import Config
from models.cnn_rnn_ctc import CNN_RNN_CTC

def create_neural_network_diagram(config):
    """Create beautiful neural network diagram with neurons"""
    print("🧠 Creating neural network diagram with neurons...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    title_text = ax.text(8, 11.5, 'Handwritten Text Recognition', 
                         fontsize=28, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', 
                                 edgecolor='#1976D2', linewidth=3))
    ax.text(8, 10.8, 'CNN + BiLSTM + CTC Architecture', 
            fontsize=16, ha='center', style='italic', color='#424242')
    
    # Draw neural network layers
    y_center = 6
    
    # INPUT LAYER with neurons
    x_pos = 1
    ax.text(x_pos + 0.5, y_center + 3, 'INPUT', fontsize=12, 
           fontweight='bold', ha='center', color='#1B5E20')
    
    # Draw input neurons
    for i in range(8):
        y = y_center + 2.5 - i * 0.5
        circle = Circle((x_pos + 0.5, y), 0.15, facecolor='#66BB6A', 
                       edgecolor='#1B5E20', linewidth=2)
        ax.add_patch(circle)
    
    ax.text(x_pos + 0.5, y_center - 2.5, f'{config.IMG_HEIGHT}×{config.IMG_WIDTH}', 
           fontsize=9, ha='center', style='italic', color='#424242')
    
    # CNN LAYERS
    cnn_layers = [
        ('CONV 32', '#42A5F5', 10),
        ('CONV 64', '#1E88E5', 12),
        ('CONV 128', '#1565C0', 14),
        ('CONV 256', '#0D47A1', 16),
    ]
    
    x_pos += 2.5
    for name, color, neurons in cnn_layers:
        # Layer box
        rect = FancyBboxPatch((x_pos, y_center - 2.5), 1.2, 5,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#0D47A1',
                             linewidth=2, alpha=0.3)
        ax.add_patch(rect)
        
        # Layer name
        ax.text(x_pos + 0.6, y_center + 3, name, fontsize=10,
               fontweight='bold', ha='center', color='white',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        # Neurons
        n_neurons = min(neurons, 12)
        for i in range(n_neurons):
            y = y_center + 2 - i * 0.35
            circle = Circle((x_pos + 0.6, y), 0.12, facecolor=color,
                          edgecolor='white', linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)
        
        # Connections to next layer
        if x_pos < 10:
            for i in range(0, n_neurons, 2):
                y1 = y_center + 2 - i * 0.35
                for j in range(0, min(neurons+2, 12), 3):
                    y2 = y_center + 2 - j * 0.35
                    ax.plot([x_pos + 0.72, x_pos + 1.98], [y1, y2],
                           color=color, alpha=0.1, linewidth=0.5)
        
        x_pos += 2.5
    
    # BiLSTM LAYER
    x_pos = 11
    
    # Forward LSTM
    rect1 = FancyBboxPatch((x_pos, y_center + 0.2), 1.8, 2,
                          boxstyle="round,pad=0.1",
                          facecolor='#EC407A', edgecolor='#880E4F',
                          linewidth=2, alpha=0.3)
    ax.add_patch(rect1)
    
    # Backward LSTM
    rect2 = FancyBboxPatch((x_pos, y_center - 2.2), 1.8, 2,
                          boxstyle="round,pad=0.1",
                          facecolor='#AB47BC', edgecolor='#4A148C',
                          linewidth=2, alpha=0.3)
    ax.add_patch(rect2)
    
    ax.text(x_pos + 0.9, y_center + 2.8, 'BiLSTM', fontsize=12,
           fontweight='bold', ha='center', color='white',
           bbox=dict(boxstyle='round', facecolor='#C2185B', alpha=0.9))
    
    # LSTM neurons
    for layer_idx, (y_start, color) in enumerate([(y_center + 0.5, '#EC407A'), 
                                                    (y_center - 1.9, '#AB47BC')]):
        for i in range(10):
            y = y_start + (1.5 - i * 0.16)
            circle = Circle((x_pos + 0.9, y), 0.1, facecolor=color,
                          edgecolor='white', linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)
    
    # Arrows showing bidirectional
    ax.annotate('', xy=(x_pos + 1.5, y_center + 1.2), xytext=(x_pos + 0.3, y_center + 1.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='#880E4F'))
    ax.annotate('', xy=(x_pos + 0.3, y_center - 1.2), xytext=(x_pos + 1.5, y_center - 1.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4A148C'))
    
    ax.text(x_pos + 0.9, y_center - 2.8, f'{config.LSTM_HIDDEN}×2', 
           fontsize=9, ha='center', style='italic', color='#424242')
    
    # OUTPUT LAYER
    x_pos = 14
    ax.text(x_pos + 0.5, y_center + 3, 'OUTPUT', fontsize=12,
           fontweight='bold', ha='center', color='#1B5E20')
    
    # Output neurons
    for i in range(8):
        y = y_center + 2.5 - i * 0.5
        circle = Circle((x_pos + 0.5, y), 0.15, facecolor='#66BB6A',
                       edgecolor='#1B5E20', linewidth=2)
        ax.add_patch(circle)
    
    ax.text(x_pos + 0.5, y_center - 2.5, f'{config.NUM_CLASSES} classes',
           fontsize=9, ha='center', style='italic', color='#424242')
    
    # Add info boxes
    info_boxes = [
        (2, 1.5, 'Feature\nExtraction', '#1976D2'),
        (7, 1.5, 'Deep\nConvolution', '#0D47A1'),
        (11.9, 1.5, 'Sequence\nModeling', '#C2185B'),
        (14.5, 1.5, 'Character\nPrediction', '#388E3C'),
    ]
    
    for x, y, text, color in info_boxes:
        ax.text(x, y, text, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                        edgecolor='white', linewidth=2, alpha=0.8),
               color='white', fontweight='bold')
    
    # Add legend in better position
    legend_x = 1
    legend_y = 0.5
    
    legend_items = [
        ('Input/Output', '#66BB6A'),
        ('CNN Layers', '#1976D2'),
        ('BiLSTM', '#C2185B'),
        ('Neurons', '#FFFFFF'),
    ]
    
    ax.text(legend_x, legend_y + 0.8, 'Legend', fontsize=11, fontweight='bold')
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y + 0.4 - i * 0.25
        if label == 'Neurons':
            circle = Circle((legend_x + 0.15, y), 0.08, facecolor='#42A5F5',
                          edgecolor='white', linewidth=1.5)
            ax.add_patch(circle)
        else:
            rect = Rectangle((legend_x, y - 0.08), 0.3, 0.16,
                           facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        ax.text(legend_x + 0.45, y, label, fontsize=9, va='center')
    
    # Add statistics
    stats_x = 12
    stats_y = 0.8
    stats_text = f"""Model Statistics
    
    Parameters: 4.6M
    Layers: 30+
    Input: {config.IMG_HEIGHT}×{config.IMG_WIDTH}
    Output: {config.NUM_CLASSES} chars
    """
    
    ax.text(stats_x, stats_y, stats_text, fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                    edgecolor='#F57F17', linewidth=2, alpha=0.9),
           verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'architecture_neural.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: architecture_neural.png")
    plt.close()

def create_3d_architecture(config):
    """Create 3D visualization of network architecture"""
    print("🎨 Creating 3D architecture visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Title
    fig.suptitle('3D Neural Network Architecture', fontsize=20, fontweight='bold')
    
    # Colors for different layer types
    colors = {
        'input': '#66BB6A',
        'conv': ['#42A5F5', '#1E88E5', '#1565C0', '#0D47A1'],
        'lstm': '#EC407A',
        'output': '#66BB6A'
    }
    
    z_pos = 0
    
    # INPUT LAYER
    layer_height = 64
    layer_width = 10
    depth = 1
    
    x = np.arange(0, layer_width, 1)
    y = np.arange(0, layer_height, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * z_pos
    
    ax.plot_surface(X, Y, Z, alpha=0.6, color=colors['input'])
    ax.text(layer_width/2, layer_height/2, z_pos-2, 'INPUT\n64×800',
           fontsize=10, ha='center', fontweight='bold')
    
    # CNN LAYERS
    z_pos += 5
    layer_sizes = [32, 64, 128, 256]
    
    for i, (size, color) in enumerate(zip(layer_sizes, colors['conv'])):
        layer_height = max(10, 64 // (2 ** (i+1)))
        layer_width = max(5, 10 // (1 if i < 2 else 1))
        
        x = np.arange(0, layer_width, 0.5)
        y = np.arange(0, layer_height, 0.5)
        X, Y = np.meshgrid(x, y)
        Z = np.ones_like(X) * z_pos
        
        ax.plot_surface(X, Y, Z, alpha=0.7, color=color)
        ax.text(layer_width/2, layer_height/2, z_pos-1, f'CONV\n{size}',
               fontsize=9, ha='center', fontweight='bold', color='white')
        
        z_pos += 3
    
    # BiLSTM LAYER
    layer_height = 20
    layer_width = 8
    
    x = np.arange(0, layer_width, 0.5)
    y = np.arange(0, layer_height, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * z_pos
    
    ax.plot_surface(X, Y, Z, alpha=0.7, color=colors['lstm'])
    ax.text(layer_width/2, layer_height/2, z_pos-1, f'BiLSTM\n{config.LSTM_HIDDEN}×2',
           fontsize=10, ha='center', fontweight='bold', color='white')
    
    z_pos += 5
    
    # OUTPUT LAYER
    layer_height = 77
    layer_width = 10
    
    x = np.arange(0, layer_width, 1)
    y = np.arange(0, layer_height, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * z_pos
    
    ax.plot_surface(X, Y, Z, alpha=0.6, color=colors['output'])
    ax.text(layer_width/2, layer_height/2, z_pos+2, f'OUTPUT\n{config.NUM_CLASSES} classes',
           fontsize=10, ha='center', fontweight='bold')
    
    # Labels
    ax.set_xlabel('Width', fontsize=10, fontweight='bold')
    ax.set_ylabel('Height', fontsize=10, fontweight='bold')
    ax.set_zlabel('Depth (Layers)', fontsize=10, fontweight='bold')
    
    # Remove grid for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'architecture_3d.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: architecture_3d.png")
    plt.close()

def create_flowchart_diagram(config):
    """Create clean flowchart-style diagram"""
    print("📊 Creating flowchart diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(6, 19, 'HTR Model Pipeline', fontsize=24, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8EAF6',
                                 edgecolor='#3F51B5', linewidth=3))
    
    y_pos = 17
    
    # Define layers with improved spacing
    layers = [
        ('Input Image', f'{config.IMG_HEIGHT}×{config.IMG_WIDTH} pixels, Grayscale', '#81C784', '📷'),
        ('Conv Block 1', '32 filters, 3×3 kernel, ReLU\nMaxPool 2×2 → 32×400', '#64B5F6', '🔲'),
        ('Conv Block 2', '64 filters, 3×3 kernel, ReLU\nMaxPool 2×2 → 16×200', '#42A5F5', '🔲'),
        ('Conv Block 3', '128 filters, 3×3 kernel, ReLU\nMaxPool 2×1 → 8×200', '#1E88E5', '🔲'),
        ('Conv Block 4', '256 filters, 3×3 kernel, ReLU\nMaxPool 2×1 → 4×200', '#1565C0', '🔲'),
        ('Reshape', 'Flatten to sequence\n200 timesteps × 1024 features', '#FFB74D', '⚙️'),
        ('BiLSTM', f'{config.LSTM_LAYERS} layers, {config.LSTM_HIDDEN} hidden units\nBidirectional processing', '#EC407A', '🔄'),
        ('Dropout', f'Rate: {config.DROPOUT}\nRegularization', '#BA68C8', '💧'),
        ('Dense Layer', f'{config.NUM_CLASSES} output units\nLog Softmax activation', '#9575CD', '📊'),
        ('CTC Loss', 'Connectionist Temporal Classification\nAlignment-free training', '#4DB6AC', '🎯'),
        ('Output', 'Predicted text sequence', '#81C784', '📝'),
    ]
    
    for i, (name, desc, color, emoji) in enumerate(layers):
        # Main box
        box_height = 1.2
        rect = FancyBboxPatch((2, y_pos - box_height/2), 8, box_height,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#263238',
                             linewidth=2.5, alpha=0.85)
        ax.add_patch(rect)
        
        # Emoji
        ax.text(2.5, y_pos, emoji, fontsize=24, ha='center', va='center')
        
        # Layer name
        ax.text(3.5, y_pos + 0.25, name, fontsize=13, fontweight='bold',
               ha='left', va='center', color='white')
        
        # Description
        ax.text(3.5, y_pos - 0.15, desc, fontsize=9, ha='left', va='center',
               color='white', style='italic')
        
        # Arrow to next layer
        if i < len(layers) - 1:
            arrow = FancyArrowPatch((6, y_pos - box_height/2 - 0.1),
                                   (6, y_pos - box_height/2 - 0.9),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='#455A64')
            ax.add_patch(arrow)
        
        y_pos -= 1.5
    
    # Add info panel
    info_text = f"""
    ⚡ Model Information
    
    • Total Parameters: 4,630,797
    • Model Size: ~18 MB
    • Inference Speed: ~50ms/image
    • Training Time: ~6-8 hours
    • Best CER: 14.60%
    • Accuracy: 64.91%
    """
    
    ax.text(6, 1.5, info_text, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF9C4',
                    edgecolor='#F57F17', linewidth=2.5, alpha=0.95),
           fontfamily='monospace', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'architecture_flowchart.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: architecture_flowchart.png")
    plt.close()

def create_compact_summary(config):
    """Create compact one-page architecture summary"""
    print("📄 Creating compact architecture summary...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('🖋️ HTR Architecture Summary', fontsize=22, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                         left=0.08, right=0.92, top=0.92, bottom=0.08)
    
    # 1. Architecture Overview (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    
    arch_boxes = [
        ('Input\n64×800', '#81C784', 1),
        ('CNN\n4 blocks', '#42A5F5', 3),
        ('BiLSTM\n2 layers', '#EC407A', 2.5),
        ('Dense\n77 classes', '#9575CD', 1.5),
        ('CTC\nLoss', '#4DB6AC', 1),
    ]
    
    x_start = 0.5
    for name, color, width in arch_boxes:
        rect = mpatches.FancyBboxPatch((x_start, 0.2), width, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='black',
                                       linewidth=2, transform=ax1.transAxes)
        ax1.add_patch(rect)
        
        ax1.text(x_start + width/2, 0.5, name, fontsize=11, fontweight='bold',
                ha='center', va='center', color='white', transform=ax1.transAxes)
        
        # Arrow
        if name != 'CTC\nLoss':
            ax1.annotate('', xy=(x_start + width + 0.15, 0.5),
                        xytext=(x_start + width + 0.05, 0.5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                        transform=ax1.transAxes)
        
        x_start += width + 0.2
    
    # 2. Layer Details (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.text(0.5, 0.95, 'Layer Details', fontsize=14, fontweight='bold',
            ha='center', transform=ax2.transAxes)
    
    layers_text = """
Conv Block 1: 32×3×3
Conv Block 2: 64×3×3
Conv Block 3: 128×3×3
Conv Block 4: 256×3×3
BiLSTM: 256 hidden
Dense: 77 outputs
    """
    
    ax2.text(0.5, 0.5, layers_text, fontsize=10, ha='center', va='center',
            transform=ax2.transAxes, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    
    # 3. Parameters (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'Parameters', fontsize=14, fontweight='bold',
            ha='center', transform=ax3.transAxes)
    
    params_data = [
        ('CNN', '388K', 8),
        ('BiLSTM', '3.9M', 85),
        ('Dense', '40K', 1),
        ('Other', '10K', 1),
    ]
    
    colors_pie = ['#42A5F5', '#EC407A', '#9575CD', '#FFB74D']
    sizes = [p[2] for p in params_data]
    
    ax3_pie = fig.add_axes([0.4, 0.35, 0.2, 0.2])
    ax3_pie.pie(sizes, labels=[p[0] for p in params_data],
               colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9, 'weight': 'bold'})
    
    # 4. Performance (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'Performance', fontsize=14, fontweight='bold',
            ha='center', transform=ax4.transAxes)
    
    perf_text = """
CER: 14.60%
WER: 35.09%
Accuracy: 64.91%

Training: ~6hrs
Inference: ~50ms
Size: ~18MB
    """
    
    ax4.text(0.5, 0.5, perf_text, fontsize=11, ha='center', va='center',
            transform=ax4.transAxes, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8),
            fontweight='bold')
    
    # 5. Data Flow (bottom, spans all columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'Data Flow & Tensor Shapes', fontsize=14, fontweight='bold',
            ha='center', transform=ax5.transAxes)
    
    flow_stages = [
        (f'(batch, 1,\n64, 800)', 0.1),
        (f'(batch, 256,\n4, 200)', 0.3),
        (f'(batch, 200,\n1024)', 0.5),
        (f'(batch, 200,\n512)', 0.7),
        (f'(batch, 200,\n77)', 0.9),
    ]
    
    for text, x in flow_stages:
        ax5.text(x, 0.5, text, fontsize=9, ha='center', va='center',
                transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='#FFF9C4',
                         edgecolor='#F57F17', linewidth=1.5))
        
        if x < 0.9:
            ax5.annotate('', xy=(x + 0.08, 0.5), xytext=(x + 0.05, 0.5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#F57F17'),
                        transform=ax5.transAxes)
    
    plt.savefig(os.path.join(config.VIS_DIR, 'architecture_summary.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: architecture_summary.png")
    plt.close()

def main():
    """Generate all architecture visualizations"""
    print("\n" + "=" * 80)
    print("🎨 ADVANCED NEURAL NETWORK ARCHITECTURE VISUALIZER")
    print("=" * 80 + "\n")
    
    config = Config()
    
    print("Creating professional architecture visualizations...\n")
    
    # Create all visualizations
    create_neural_network_diagram(config)
    create_3d_architecture(config)
    create_flowchart_diagram(config)
    create_compact_summary(config)
    
    print("\n" + "=" * 80)
    print("✅ ALL ARCHITECTURE VISUALIZATIONS CREATED!")
    print("=" * 80)
    print(f"\n📁 Location: {config.VIS_DIR}")
    print("\n🎨 Generated files:")
    print("   1. architecture_neural.png       - Beautiful neural network with neurons")
    print("   2. architecture_3d.png           - 3D visualization of layers")
    print("   3. architecture_flowchart.png    - Detailed flowchart with emojis")
    print("   4. architecture_summary.png      - Compact one-page summary")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
