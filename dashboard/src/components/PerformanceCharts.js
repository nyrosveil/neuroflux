import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './PerformanceCharts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function PerformanceCharts({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="performance-charts loading">
        <h3>📊 Performance Charts</h3>
        <p>Loading performance data...</p>
      </div>
    );
  }

  // Prepare data for charts
  const labels = data.map((_, index) => `Cycle ${index + 1}`);
  const executionTimes = data.map(agent => agent.execution_time || 0);
  const successRates = data.map(agent => agent.success ? 1 : 0);

  const executionTimeData = {
    labels,
    datasets: [
      {
        label: 'Execution Time (s)',
        data: executionTimes,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
    ],
  };

  const successRateData = {
    labels,
    datasets: [
      {
        label: 'Success Rate',
        data: successRates,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Agent Performance Metrics',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="performance-charts">
      <h3>📊 Performance Charts</h3>

      <div className="charts-container">
        <div className="chart-wrapper">
          <h4>Execution Time</h4>
          <Line data={executionTimeData} options={options} />
        </div>

        <div className="chart-wrapper">
          <h4>Success Rate</h4>
          <Line data={successRateData} options={options} />
        </div>
      </div>

      <div className="performance-summary">
        <div className="summary-item">
          <span className="label">Average Execution Time</span>
          <span className="value">
            {(executionTimes.reduce((a, b) => a + b, 0) / executionTimes.length).toFixed(2)}s
          </span>
        </div>

        <div className="summary-item">
          <span className="label">Overall Success Rate</span>
          <span className="value">
            {((successRates.reduce((a, b) => a + b, 0) / successRates.length) * 100).toFixed(1)}%
          </span>
        </div>

        <div className="summary-item">
          <span className="label">Total Cycles</span>
          <span className="value">{data.length}</span>
        </div>
      </div>
    </div>
  );
}

export default PerformanceCharts;