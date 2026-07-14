document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const spinner = submitBtn.querySelector('.spinner');
    const btnText = submitBtn.querySelector('.btn-text');
    
    // Status Stage Indicators
    const pipelineStatus = document.getElementById('pipeline-status');
    const statusText = document.getElementById('status-text');
    
    // Result Elements
    const resultOutput = document.getElementById('result-output');
    const sparqlCode = document.getElementById('sparql-code');
    const tableContainer = document.getElementById('table-container');
    const copyBtn = document.getElementById('copy-btn');

    let loaderIntervals = [];

    // Form submission handler
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = queryInput.value.trim();
        if (!question) return;
        
        await executeQuery(question);
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(sparqlCode.textContent);
        copyBtn.textContent = "Copied!";
        setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
    });

    // Execute pipeline call
    async function executeQuery(question) {
        // Reset states
        submitBtn.disabled = true;
        spinner.hidden = false;
        btnText.textContent = "Search";
        resultOutput.hidden = true;
        pipelineStatus.hidden = false;
        
        startStatusSimulation();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            stopStatusSimulation();

            if (response.ok) {
                const data = await response.json();
                pipelineStatus.hidden = true;
                submitBtn.disabled = false;
                spinner.hidden = true;
                
                if (data.success && data.result) {
                    renderResults(data.result);
                } else {
                    renderFailure(data.result?.error || "Pipeline failed to resolve query.");
                }
            } else {
                const errData = await response.json();
                renderFailure(errData.error || response.statusText);
            }
        } catch (e) {
            stopStatusSimulation();
            renderFailure("Connection Error: " + e.message);
        }
    }

    // Simulate current pipeline stage on screen
    function startStatusSimulation() {
        statusText.textContent = "Extracting entities & properties...";
        
        const i1 = setTimeout(() => {
            statusText.textContent = "Linking candidate IDs on Wikidata...";
        }, 1500);

        const i2 = setTimeout(() => {
            statusText.textContent = "Generating SPARQL query...";
        }, 3200);

        const i3 = setTimeout(() => {
            statusText.textContent = "Running query on Wikidata...";
        }, 5000);

        loaderIntervals = [i1, i2, i3];
    }

    function stopStatusSimulation() {
        loaderIntervals.forEach(clearTimeout);
    }

    // Render results in table
    function renderResults(result) {
        resultOutput.hidden = false;
        
        // Render SPARQL
        sparqlCode.textContent = result.sparql || "";
        
        // Render Table
        const executionResults = result.results;
        if (!executionResults || !executionResults.columns || executionResults.columns.length === 0) {
            tableContainer.innerHTML = '<div class="empty-state">No headers returned.</div>';
            return;
        }
        
        const cols = executionResults.columns;
        const rows = executionResults.rows;

        if (!rows || rows.length === 0) {
            tableContainer.innerHTML = '<div class="empty-state">No matching rows found on Wikidata.</div>';
            return;
        }

        let html = '<table class="results-table"><thead><tr>';
        cols.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead><tbody>';

        rows.forEach(row => {
            html += '<tr>';
            cols.forEach(col => {
                const val = row[col] !== undefined ? row[col] : 'N/A';
                if (typeof val === 'string' && val.startsWith('http')) {
                    const label = val.split('/').pop();
                    html += `<td><a href="${val}" target="_blank" class="uri-link">${label}</a></td>`;
                } else {
                    html += `<td>${val}</td>`;
                }
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        tableContainer.innerHTML = html;
    }

    // Render pipeline failures
    function renderFailure(errorMsg) {
        pipelineStatus.hidden = true;
        submitBtn.disabled = false;
        spinner.hidden = true;
        resultOutput.hidden = false;
        
        sparqlCode.textContent = "-- No SPARQL generated --";
        tableContainer.innerHTML = `
            <div class="empty-state" style="color: var(--text-muted)">
                <strong>Error running pipeline:</strong><br>
                ${errorMsg}
            </div>
        `;
    }
});
