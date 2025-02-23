import torch
import torch_directml
import numpy as np
from pointnet_modified import PointNet
from dataset import PointCloudDataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import datetime

def save_plt_to_img():
    """Save current matplotlib figure to bytes buffer"""
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    plt.close()
    return img_buffer

def evaluate_model():
    # Create PDF document with A4 portrait
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = SimpleDocTemplate(
        f"model_evaluation_{timestamp}.pdf",
        pagesize=(8.27 * inch, 11.69 * inch),  # A4 size
        rightMargin=0.5*inch, 
        leftMargin=0.5*inch,
        topMargin=0.5*inch, 
        bottomMargin=0.5*inch
    )
    
    # Adjust figure sizes for A4
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (7, 7)  # Square aspect ratio for confusion matrix
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 8
    
    # Setup styles and elements
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Point Cloud Log Counter - Model Evaluation", 
        ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30)))
    elements.append(Paragraph(
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
        styles['Normal']))
    elements.append(Spacer(1, 20))

    # Add model architecture details
    arch_elements = []
    arch_elements.append(Paragraph("Model Architecture and Training Details", styles['Heading2']))
    arch_elements.append(Spacer(1, 10))
    
    # Get model architecture as string
    with open('pointnet_modified.py', 'r') as f:
        model_code = f.read()
    
    # Add dataset info
    device = torch_directml.device()
    dataset = PointCloudDataset("C:/output", device=device)
    model = PointNet(num_classes=21).to(device)
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_samples = len(dataset)
    with torch.no_grad():
        for idx in range(total_samples):
            points, target = dataset[idx]
            points = points.unsqueeze(0)
            
            output = model(points)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()
            confidence = probs[pred].item() * 100
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_samples} samples")
    
    # Add model architecture code
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=7,
        fontName='Courier',
        spaceAfter=8,
        spaceBefore=8,
        backColor=colors.lightgrey,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=5
    )
    
    arch_elements.append(Paragraph("Model Architecture Code:", styles['Heading3']))
    arch_elements.append(Spacer(1, 5))
    arch_elements.append(Paragraph(model_code, code_style))
    
    # Add to document at the beginning after title
    elements.insert(3, KeepTogether(arch_elements))
    elements.insert(4, Spacer(1, 20))

    # Evaluate model
    print("\nEvaluating model...")
    accuracy_by_count = defaultdict(list)
    confidence_by_count = defaultdict(list)
    
    for idx in range(total_samples):
        points, target = dataset[idx]
        points = points.unsqueeze(0)
        
        output = model(points)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        pred = output.argmax(1).item()
        confidence = probs[pred].item() * 100
        
        accuracy_by_count[target.item()].append(pred == target.item())
        confidence_by_count[target.item()].append(confidence)
    
    # Calculate metrics
    all_predictions = np.array(list(accuracy_by_count.values())[0])
    all_targets = np.array(list(accuracy_by_count.keys()))
    overall_accuracy = (all_predictions == all_targets).mean() * 100
    mae = np.abs(all_predictions - all_targets).mean()
    
    per_class_accuracy = {}
    per_class_confidence = {}
    per_class_samples = {}
    
    for count in sorted(accuracy_by_count.keys()):
        accuracies = accuracy_by_count[count]
        confidences = confidence_by_count[count]
        per_class_accuracy[count] = np.mean(accuracies) * 100
        per_class_confidence[count] = np.mean(confidences)
        per_class_samples[count] = len(accuracies)
    
    # Add metrics to PDF
    metrics_elements = []
    metrics_elements.append(Paragraph("Overall Metrics", styles['Heading2']))
    metrics_elements.append(Spacer(1, 10))
    metrics_data = [
        ["Metric", "Value"],
        ["Overall Accuracy", f"{overall_accuracy:.2f}%"],
        ["Mean Absolute Error", f"{mae:.2f} logs"]
    ]
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    metrics_elements.append(metrics_table)
    elements.append(KeepTogether(metrics_elements))
    elements.append(Spacer(1, 20))

    # Add confusion matrix with adjusted size
    confusion_elements = []
    confusion_elements.append(Paragraph("Confusion Matrix", styles['Heading2']))
    confusion_elements.append(Spacer(1, 10))
    
    plt.figure()  # Uses default size set above
    cm = confusion_matrix(all_targets, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 21),
                yticklabels=range(1, 21),
                annot_kws={'size': 6})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Number of Logs')
    plt.ylabel('Actual Number of Logs')
    confusion_elements.append(Image(save_plt_to_img(), width=7*inch, height=7*inch))
    elements.append(KeepTogether(confusion_elements))
    elements.append(Spacer(1, 20))

    # Add accuracy plot with adjusted size
    accuracy_elements = []
    accuracy_elements.append(Paragraph("Accuracy by Log Count", styles['Heading2']))
    accuracy_elements.append(Spacer(1, 10))
    
    plt.figure(figsize=(7, 3.5))  # Wider than tall for bar chart
    counts = sorted(per_class_accuracy.keys())
    plt.bar(counts, [per_class_accuracy[c] for c in counts])
    plt.axhline(y=overall_accuracy, color='r', linestyle='--', 
                label=f'Overall Accuracy: {overall_accuracy:.1f}%')
    plt.title('Accuracy by Log Count')
    plt.xlabel('Number of Logs')
    plt.ylabel('Accuracy (%)')
    plt.legend(fontsize=8)
    accuracy_elements.append(Image(save_plt_to_img(), width=7*inch, height=3.5*inch))
    elements.append(KeepTogether(accuracy_elements))
    elements.append(Spacer(1, 20))

    # Add per-class results
    results_elements = []
    results_elements.append(Paragraph("Per-class Results", styles['Heading2']))
    results_elements.append(Spacer(1, 10))
    per_class_data = [["Log Count", "Accuracy", "Confidence", "Samples"]]
    for count in sorted(per_class_accuracy.keys()):
        per_class_data.append([
            str(count),
            f"{per_class_accuracy[count]:.2f}%",
            f"{per_class_confidence[count]:.2f}%",
            str(per_class_samples[count])
        ])
    
    per_class_table = Table(per_class_data)
    per_class_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    results_elements.append(per_class_table)
    elements.append(KeepTogether(results_elements))

    # Build PDF
    doc.build(elements)
    print(f"\nPDF report generated: model_evaluation_{timestamp}.pdf")

def main():
    print("\nStarting model evaluation...")
    evaluate_model()  # This generates the PDF report

if __name__ == "__main__":
    main() 