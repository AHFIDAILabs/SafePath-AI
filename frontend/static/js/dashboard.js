document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const loader = document.getElementById('loader');
    
    let riskChart = null;
    let protectiveChart = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        console.log('Form submitted, starting prediction...');
        
        // Show loader and hide previous results
        loader.style.display = 'block';
        resultsContainer.style.display = 'block';
        resultsContent.style.display = 'none';
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyzing...';

        // Collect form data
        const formData = new FormData(form);
        const payload = {};
        
        // Handle all form fields properly
        for (let [key, value] of formData.entries()) {
            const element = form.elements[key];
            if (element.type === 'checkbox') {
                payload[key] = element.checked ? 1 : 0;
            } else if (element.type === 'number') {
                payload[key] = parseInt(value, 10);
            } else {
                payload[key] = value;
            }
        }
        
        // Ensure all checkbox fields are included (unchecked boxes won't be in FormData)
        const checkboxFields = [
            'PLWD', 'PLHIV', 'IDP', 'drug_user', 'widow', 'out_of_school_child',
            'minor', 'household_help', 'child_apprentice', 'orphans', 'female_sex_worker'
        ];
        
        checkboxFields.forEach(field => {
            if (!(field in payload)) {
                payload[field] = 0;
            }
        });

        console.log('Payload being sent:', payload);

        try {
            const response = await fetch('http://127.0.0.1:8000/api/v1/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Response data:', data);
            displayResults(data);

        } catch (error) {
            console.error('Error occurred:', error);
            resultsContent.innerHTML = `<div class="error">
                <h3>Error</h3>
                <p>${error.message}</p>
                <p>Please check that the backend server is running on port 8000.</p>
            </div>`;
        } finally {
            // Hide loader and show results content
            loader.style.display = 'none';
            resultsContent.style.display = 'block';
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Risk Assessment';
        }
    });

    function displayResults(data) {
        document.getElementById('prediction-text').textContent = data.prediction;
        document.getElementById('prediction-text').className = data.prediction === 'High Risk' ? 'risk-high' : 'risk-low';
        
        document.getElementById('probability-text').textContent = `${(data.risk_probability * 100).toFixed(1)}%`;
        document.getElementById('confidence-text').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        
        // Display comprehensive summary with proper formatting
        displayComprehensiveSummary(data.generative_summary);

        // Create charts
        createCharts(data.key_risk_factors, data.key_protective_factors);
    }

    function displayComprehensiveSummary(summaryText) {
        const summaryContainer = document.getElementById('generative-summary');
        
        // Parse the summary to separate assessment and recommendations
        const parts = summaryText.split('RECOMMENDATIONS:');
        const assessment = parts[0].replace('ASSESSMENT SUMMARY:', '').trim();
        const recommendations = parts.length > 1 ? parts[1].trim() : '';
        
        // Create formatted HTML
        let formattedHTML = '';
        
        if (assessment) {
            formattedHTML += `
                <div class="summary-section">
                    <h4 style="color: #18bc9c; margin-bottom: 0.5rem; font-size: 1.1rem;">
                        <i class="fas fa-chart-line"></i> Assessment Summary
                    </h4>
                    <p style="line-height: 1.6; margin-bottom: 1rem;">${assessment}</p>
                </div>
            `;
        }
        
        if (recommendations) {
            // Format recommendations as a list if they're numbered
            let recommendationHTML = recommendations;
            if (recommendations.includes('1)')) {
                // Split numbered recommendations and format as list
                const recItems = recommendations.split(/\d+\)\s/).filter(item => item.trim());
                recommendationHTML = '<ul style="padding-left: 1.2rem; line-height: 1.6;">' +
                    recItems.map(item => `<li style="margin-bottom: 0.5rem;">${item.trim()}</li>`).join('') +
                    '</ul>';
            } else {
                recommendationHTML = `<p style="line-height: 1.6;">${recommendations}</p>`;
            }
            
            formattedHTML += `
                <div class="recommendations-section">
                    <h4 style="color: #18bc9c; margin-bottom: 0.5rem; font-size: 1.1rem;">
                        <i class="fas fa-lightbulb"></i> Recommendations
                    </h4>
                    ${recommendationHTML}
                </div>
            `;
        }
        
        // If no structured format found, display as is
        if (!assessment && !recommendations) {
            formattedHTML = `<p style="line-height: 1.6;">${summaryText}</p>`;
        }
        
        summaryContainer.innerHTML = formattedHTML;
    }

    function createCharts(riskFactors, protectiveFactors) {
        // Destroy existing charts if they exist
        if (riskChart) riskChart.destroy();
        if (protectiveChart) protectiveChart.destroy();

        // Risk Factors Chart
        if (riskFactors && riskFactors.length > 0) {
            const riskCtx = document.getElementById('risk-factors-chart').getContext('2d');
            riskChart = new Chart(riskCtx, {
                type: 'bar',
                data: {
                    labels: riskFactors.map(f => formatFeatureName(f.feature)),
                    datasets: [{
                        label: 'Impact on Risk (SHAP Value)',
                        data: riskFactors.map(f => f.impact),
                        backgroundColor: 'rgba(255, 107, 107, 0.6)',
                        borderColor: 'rgba(255, 107, 107, 1)',
                        borderWidth: 1
                    }]
                },
                options: { 
                    indexAxis: 'y', 
                    responsive: true, 
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
        }

        // Protective Factors Chart
        if (protectiveFactors && protectiveFactors.length > 0) {
            const protectiveCtx = document.getElementById('protective-factors-chart').getContext('2d');
            protectiveChart = new Chart(protectiveCtx, {
                type: 'bar',
                data: {
                    labels: protectiveFactors.map(f => formatFeatureName(f.feature)),
                    datasets: [{
                        label: 'Impact on Protection (SHAP Value)',
                        data: protectiveFactors.map(f => Math.abs(f.impact)), // Show as positive for better visualization
                        backgroundColor: 'rgba(81, 207, 102, 0.6)',
                        borderColor: 'rgba(81, 207, 102, 1)',
                        borderWidth: 1
                    }]
                },
                options: { 
                    indexAxis: 'y', 
                    responsive: true, 
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
        }
    }

    function formatFeatureName(featureName) {
        // Convert feature names to readable format
        const nameMap = {
            'economic_dependency_score': 'Economic Dependency',
            'survivor_sex': 'Gender',
            'survivor_age': 'Age',
            'who_survivor/victim_stay_with': 'Living Arrangement',
            'income_stability_score': 'Income Stability',
            'housing_security_score': 'Housing Security',
            'social_isolation_score': 'Social Isolation',
            'employment_status_victim_main': 'Employment Status',
            'educational_status': 'Education Level',
            'community_connection_score': 'Community Connections',
            'marital_status': 'Marital Status',
            'financial_access_proxy': 'Financial Access'
        };
        
        return nameMap[featureName] || featureName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
});