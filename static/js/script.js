document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    // Update result display
    document.getElementById('prediction').textContent = result.prediction;
    document.getElementById('probability').textContent = result.probability;
    document.getElementById('accuracy').textContent = result.metrics.accuracy;
    document.getElementById('precision').textContent = result.metrics.precision;
    document.getElementById('recall').textContent = result.metrics.recall;
    document.getElementById('f1').textContent = result.metrics.f1;
    document.getElementById('feature-importance').src = result.feature_importance;
    document.getElementById('confusion-matrix').src = result.confusion_matrix;
    
    document.getElementById('result').style.display = 'block';
});