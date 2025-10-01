// NEW: Tab switching function
function showTab(tabName) {
    // Hide all tab content panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    // Deactivate all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show the selected tab content
    const activePane = document.getElementById(tabName + '-section');
    if (activePane) {
        activePane.classList.add('active');
    }

    // Activate the selected tab button
    const activeButton = document.querySelector(`.tab-btn[onclick="showTab('${tabName}')"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const loader = document.getElementById('loader');

    // Clear any URL parameters to prevent resubmission on refresh
    if (window.location.search) {
        if (window.history && window.history.replaceState) {
            window.history.replaceState({}, document.title, window.location.pathname);
        }
    }

    if (!form) {
        console.error('Form element not found');
        return;
    }

    let waterfallChart = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loader and results container
        if (loader) loader.style.display = 'block';
        if (resultsContainer) resultsContainer.style.display = 'block';
        if (resultsContent) resultsContent.style.display = 'none';
        if (predictBtn) {
            predictBtn.disabled = true;
            predictBtn.textContent = 'Analyzing...';
        }
        
        // Scroll to results for better user experience
        resultsContainer.scrollIntoView({ behavior: 'smooth' });

        // Collect form data, including unchecked checkboxes
        const formData = new FormData(form);
        const payload = {};
        for (let [key, value] of formData.entries()) {
            const element = form.elements[key];
            if (element && element.type === 'checkbox') {
                payload[key] = element.checked ? 1 : 0;
            } else if (element && element.type === 'number') {
                payload[key] = parseInt(value, 10);
            } else {
                payload[key] = value;
            }
        }
        const checkboxFields = [
            'PLWD', 'PLHIV', 'IDP', 'drug_user', 'widow', 'out_of_school_child',
            'minor', 'household_help', 'child_apprentice', 'orphans', 'female_sex_worker'
        ];
        checkboxFields.forEach(field => {
            if (!(field in payload)) {
                payload[field] = 0;
            }
        });

        try {
            const response = await fetch('https://predict-gbv-risk.onrender.com/api/v1/predict', { //'http://127.0.0.1:8000/api/v1/predict' -> for local testing
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            displayResults(data, payload);
        } catch (error) {
            showError(error.message);
        } finally {
            if (loader) loader.style.display = 'none';
            if (resultsContent) resultsContent.style.display = 'block';
            if (predictBtn) {
                predictBtn.disabled = false;
                predictBtn.textContent = 'Predict Risk Assessment';
            }
        }
    });


    function showError(message) {
        const errorHTML = `
            <div class="error">
                <h3><i class="fas fa-exclamation-triangle"></i> An Error Occurred</h3>
                <p>${message}</p>
                <p>Please ensure the backend server is running and accessible.</p>
            </div>
        `;
        if (resultsContent) {
            resultsContent.innerHTML = errorHTML;
        } else if (resultsContainer) {
            resultsContainer.innerHTML = errorHTML;
            resultsContainer.style.display = 'block';
        }
    }

    function displayResults(data, originalPayload) {
        // --- Populate Summary Cards ---
        const predictionText = document.getElementById('prediction-text');
        const probabilityText = document.getElementById('probability-text');
        const confidenceText = document.getElementById('confidence-text');

        if (predictionText) {
            predictionText.textContent = data.prediction;
            predictionText.className = data.prediction === 'High Risk' ? 'risk-high' : 'risk-low';
        }
        if (probabilityText) {
            probabilityText.textContent = `${(data.risk_probability * 100).toFixed(1)}%`;
        }
        if (confidenceText) {
            confidenceText.textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A';
        }

        // --- Populate Data-Driven Insights Tab ---
        let summaryToDisplay = data.generative_summary;
        if (!summaryToDisplay || summaryToDisplay.trim() === "" || summaryToDisplay === "No summary available.") {
            summaryToDisplay = createEnhancedFallbackSummary(data, originalPayload);
        }
        displayGenerativeSummary(summaryToDisplay);

        // --- Populate Key Drivers Tab ---
        if (data.key_risk_factors && data.key_protective_factors) {
            createCharts(data.key_risk_factors, data.key_protective_factors);
        }

        // --- Populate Model Input Tab ---
        displayModelInputs(data.processed_features);
        
        // --- Reset to the first tab ---
        showTab('insights');
    }

    function displayGenerativeSummary(summaryText) {
        const assessmentList = document.getElementById('assessment-summary');
        const recommendationsList = document.getElementById('recommendations-list');
        if (!assessmentList || !recommendationsList) return;

        assessmentList.innerHTML = '';
        recommendationsList.innerHTML = '';

        if (!summaryText || summaryText.trim() === "" || summaryText === "No summary available.") {
            assessmentList.innerHTML = '<li>Assessment analysis is being processed.</li>';
            recommendationsList.innerHTML = '<li>Personalized recommendations will be generated shortly.</li>';
            return;
        }

        const parts = summaryText.split('RECOMMENDATIONS:');
        const assessment = parts[0].replace('ASSESSMENT SUMMARY:', '').trim();
        const recommendations = parts.length > 1 ? parts[1].trim() : '';

        if (assessment && assessment.length > 10) {
            const assessmentSentences = assessment.split(/(?<!\d)\.(?!\d)/).filter(item => item.trim().length > 5);
            assessmentList.innerHTML = assessmentSentences.map(sentence => `<li>${sentence.trim()}.</li>`).join('');
        }

        if (recommendations && recommendations.length > 10) {
            let recItems = recommendations.split(/\d+[\)\.]\s*/).filter(item => item.trim().length > 5);
            if (recItems.length <= 1) recItems = recommendations.split(/[-•]\s*/).filter(item => item.trim().length > 5);
            if (recItems.length <= 1) recItems = recommendations.split(/[.!?]+/).filter(item => item.trim().length > 10);
            recommendationsList.innerHTML = recItems.map(item => `<li>${item.trim().replace(/[.!?]*$/, '')}.</li>`).join('');
        }
    }

    function createEnhancedFallbackSummary(data, originalPayload) {
        const riskLevel = data.prediction;
        const probability = (data.risk_probability * 100).toFixed(1);
        const age = originalPayload.survivor_age || 'unknown';
        const gender = originalPayload.survivor_sex || 'unknown';
        let assessment = `ASSESSMENT SUMMARY: This ${age}-year-old ${gender.toLowerCase()} individual has been assessed as ${riskLevel} with a ${probability}% probability. This assessment provides a foundation for targeted interventions.`;
        let recommendations = "RECOMMENDATIONS: 1) Conduct a detailed psychosocial assessment to understand specific needs. 2) Establish regular, supportive monitoring. 3) Coordinate with multidisciplinary support services for comprehensive care.";
        return `${assessment}\n\n${recommendations}`;
    }

    function displayModelInputs(featuresData) {
        const container = document.getElementById('model-features');
        if (!container) return;
        const engineeredFeatures = [
            'economic_dependency_score', 'survivor_sex', 'survivor_age', 'who_survivor/victim_stay_with',
            'income_stability_score', 'housing_security_score', 'social_isolation_score',
            'employment_status_victim_main', 'educational_status', 'community_connection_score',
            'marital_status', 'financial_access_proxy'
        ];
        const featureDisplayNames = {
            'economic_dependency_score': 'Economic Dependency', 'survivor_sex': 'Sex', 'survivor_age': 'Age',
            'who_survivor/victim_stay_with': 'Living Arrangement', 'income_stability_score': 'Income Stability',
            'housing_security_score': 'Housing Security', 'social_isolation_score': 'Social Isolation',
            'employment_status_victim_main': 'Employment Status', 'educational_status': 'Educational Status',
            'community_connection_score': 'Community Connection', 'marital_status': 'Marital Status',
            'financial_access_proxy': 'Financial Access'
        };
        let textContent = '';
        engineeredFeatures.forEach(featureName => {
            let displayValue = featuresData && featuresData.hasOwnProperty(featureName) ? featuresData[featureName] : 'N/A';
            textContent += `${featureDisplayNames[featureName].padEnd(30, ' ')}: ${displayValue}\n`;
        });
        container.textContent = textContent.trim();
    }

    function createCharts(riskFactors, protectiveFactors) {
        const chartCanvas = document.getElementById('waterfall-chart');
        if (!chartCanvas) return;

        if (waterfallChart) waterfallChart.destroy(); // Destroy previous chart instance

        const ctx = chartCanvas.getContext('2d');
        const riskData = riskFactors.map(f => ({ feature: formatFeatureName(f.feature), value: f.impact, type: 'risk' }));
        const protectiveData = protectiveFactors.map(f => ({ feature: formatFeatureName(f.feature), value: Math.abs(f.impact), type: 'protective' }));
        const allData = [...riskData, ...protectiveData].sort((a, b) => b.value - a.value);
        
        waterfallChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: allData.map(d => d.feature),
                datasets: [{
                    label: 'Feature Impact',
                    data: allData.map(d => d.type === 'risk' ? d.value : -d.value),
                    backgroundColor: allData.map(d => d.type === 'risk' ? 'rgba(255, 107, 107, 0.8)' : 'rgba(81, 207, 102, 0.8)')
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    title: { display: false },
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Impact: ${Math.abs(context.raw).toFixed(3)} (${context.raw > 0 ? 'Increases Risk' : 'Decreases Risk'})`
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: '#fff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    y: { ticks: { color: '#fff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });

        generateChartDescription(riskData, protectiveData);
    }

    function generateChartDescription(riskData, protectiveData) {
        const container = document.querySelector('.chart-description');
        if (!container) return;

        let descriptionText = "The chart visualizes how each factor influences the prediction. ";
        if (riskData && riskData.length > 0) {
            const topRisk = riskData.sort((a, b) => b.value - a.value).slice(0, 3).map(f => f.feature).join(', ');
            descriptionText += `Factors like <strong>${topRisk}</strong> push the prediction towards higher risk (bars to the right). `;
        }
        if (protectiveData && protectiveData.length > 0) {
            const topProtective = protectiveData.sort((a, b) => b.value - a.value).slice(0, 3).map(f => f.feature).join(', ');
            descriptionText += `Conversely, factors such as <strong>${topProtective}</strong> reduce the predicted risk (bars to the left).`;
        }
        container.innerHTML = `<strong>Interpretation:</strong> ${descriptionText}`;
    }

    function formatFeatureName(featureName) {
        const nameMap = {
            'economic_dependency_score': 'Economic Dependency', 'survivor_sex': 'Gender', 'survivor_age': 'Age',
            'who_survivor/victim_stay_with': 'Living Arrangement', 'income_stability_score': 'Income Stability',
            'housing_security_score': 'Housing Security', 'social_isolation_score': 'Social Isolation',
            'employment_status_victim_main': 'Employment Status', 'educational_status': 'Education Level',
            'community_connection_score': 'Community Connections', 'marital_status': 'Marital Status',
            'financial_access_proxy': 'Financial Access'
        };
        return nameMap[featureName] || featureName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
});

// Clear cache on window unload to ensure fresh data
window.addEventListener('beforeunload', () => {
    if ('caches' in window) {
        caches.keys().then(names => {
            names.forEach(name => caches.delete(name));
        });
    }
});