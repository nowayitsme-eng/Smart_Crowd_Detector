// Zaytrics - Crowd Monitoring Dashboard

class ZaytricsDashboard {
    constructor() {
        this.isRunning = false;
        this.currentSource = 'camera';
        this.peopleCount = 0;
        this.fps = 0;
        this.init();
    }

    init() {
        console.log('Zaytrics Dashboard Initialized');
        this.setupEventListeners();
        this.setupFileUpload();
        this.startStatsPolling();
        this.animateStats();
    }

    setupEventListeners() {
        // Control button interactions
        document.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (!e.currentTarget.id.includes('heatmap')) {
                    document.querySelectorAll('.control-btn').forEach(b => {
                        if (!b.id.includes('heatmap')) {
                            b.classList.remove('active');
                        }
                    });
                    e.currentTarget.classList.add('active');
                }
            });
        });
    }

    setupFileUpload() {
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const uploadStatus = document.getElementById('uploadStatus');

        // Click to upload
        uploadZone.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadVideo(file);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = 'var(--primary)';
            uploadZone.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
            uploadZone.style.backgroundColor = '';
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
            uploadZone.style.backgroundColor = '';
            
            const file = e.dataTransfer.files[0];
            if (file) {
                this.uploadVideo(file);
            }
        });
    }

    async uploadVideo(file) {
        const uploadStatus = document.getElementById('uploadStatus');
        const loopVideo = document.getElementById('loopVideo').checked;

        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        if (!allowedTypes.includes(file.type)) {
            uploadStatus.innerHTML = '<div class="error">❌ Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM.</div>';
            return;
        }

        // Validate file size (100MB)
        if (file.size > 100 * 1024 * 1024) {
            uploadStatus.innerHTML = '<div class="error">❌ File too large. Maximum size is 100MB.</div>';
            return;
        }

        uploadStatus.innerHTML = '<div class="info">⏳ Uploading video...</div>';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('loop', loopVideo);

        try {
            const response = await fetch('/api/upload_video', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                uploadStatus.innerHTML = '<div class="success">✅ Video uploaded successfully!</div>';
                this.currentSource = 'video';
                
                // Auto-switch to uploaded video
                setTimeout(() => {
                    uploadStatus.innerHTML = '';
                }, 3000);
            } else {
                uploadStatus.innerHTML = `<div class="error">❌ ${data.error}</div>`;
            }
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.innerHTML = '<div class="error">❌ Upload failed. Please try again.</div>';
        }
    }

    startStatsPolling() {
        // Clear any existing interval
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
        }
        
        // Poll stats from Flask API
        this.statsInterval = setInterval(async () => {
            if (this.isRunning) {
                await this.updateStatsFromAPI();
            }
        }, 1000);
    }
    
    stopStatsPolling() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
    }

    async updateStatsFromAPI() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            this.peopleCount = data.count || 0;
            this.fps = data.fps || 0;
            
            document.getElementById('peopleCount').textContent = this.peopleCount;
            document.getElementById('fpsCount').textContent = this.fps.toFixed(1);
            
            // Update status indicator based on alert level
            const statusDot = document.querySelector('.status-dot');
            if (statusDot) {
                const alertLevel = data.alert_level || 'normal';
                statusDot.className = 'status-dot';
                if (alertLevel === 'warning') {
                    statusDot.style.backgroundColor = '#f59e0b';
                } else if (alertLevel === 'critical') {
                    statusDot.style.backgroundColor = '#ef4444';
                } else {
                    statusDot.style.backgroundColor = '#10b981';
                }
            }
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    }

    animateStats() {
        // Animate stat numbers when they change
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'characterData' || mutation.type === 'childList') {
                    const element = mutation.target.parentElement;
                    if (element && element.classList.contains('stat-value')) {
                        element.style.transform = 'scale(1.1)';
                        setTimeout(() => {
                            element.style.transform = 'scale(1)';
                        }, 300);
                    }
                }
            });
        });

        const statElements = document.querySelectorAll('.stat-value');
        statElements.forEach(element => {
            observer.observe(element, {
                characterData: true,
                childList: true,
                subtree: true
            });
        });
    }
}

// Global functions for button handlers
let dashboard;

async function switchSource(source) {
    const uploadSection = document.getElementById('uploadSection');
    const liveCameraBtn = document.getElementById('liveCameraBtn');
    const uploadVideoBtn = document.getElementById('uploadVideoBtn');
    
    if (source === 'upload') {
        uploadSection.style.display = 'block';
        dashboard.currentSource = 'upload';
    } else {
        uploadSection.style.display = 'none';
        dashboard.currentSource = 'camera';
        
        // Switch backend to camera
        console.log('Switching to camera source...');
        try {
            const response = await fetch('/api/switch_source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_type: 'camera' })
            });
            const data = await response.json();
            console.log('Switched to camera:', data);
        } catch (err) {
            console.error('Error switching source:', err);
        }
    }
}

async function startMonitoring() {
    console.log('Starting monitoring...');
    dashboard.isRunning = true;
    
    // Restart stats polling
    if (dashboard) {
        dashboard.startStatsPolling();
    }
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const placeholder = document.getElementById('videoPlaceholder');
    const videoFeed = document.getElementById('videoFeed');
    
    startBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    
    try {
        // First call the start API to set state
        const response = await fetch('/api/start', { method: 'POST' });
        const data = await response.json();
        console.log('Start API response:', data);
        
        if (data.status === 'started') {
            // Small delay to let backend initialize
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Now set video source and display
            if (videoFeed) {
                console.log('Setting video feed source...');
                videoFeed.src = '/video_feed?t=' + new Date().getTime();
                
                // Add load event listener for debugging
                videoFeed.onload = function() {
                    console.log('Video feed loaded successfully');
                };
                
                videoFeed.onerror = function(e) {
                    console.error('Video feed error:', e);
                    alert('Failed to load video stream. Check console for details.');
                };
                
                videoFeed.style.display = 'block';
                console.log('Video feed displayed');
            }
            
            // Hide placeholder
            if (placeholder) {
                placeholder.style.display = 'none';
                console.log('Placeholder hidden');
            }
        }
    } catch (error) {
        console.error('Error starting monitoring:', error);
        alert('Failed to start monitoring. Error: ' + error.message);
        dashboard.isRunning = false;
        startBtn.style.display = 'block';
        stopBtn.style.display = 'none';
    }
}

async function stopMonitoring() {
    console.log('Stopping monitoring...');
    dashboard.isRunning = false;
    
    // Stop stats polling to prevent unnecessary API calls
    if (dashboard) {
        dashboard.stopStatsPolling();
    }
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    startBtn.style.display = 'block';
    stopBtn.style.display = 'none';
    
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'stopped') {
            // Hide video feed
            const placeholder = document.getElementById('videoPlaceholder');
            const videoFeed = document.getElementById('videoFeed');
            
            if (videoFeed) {
                videoFeed.style.display = 'none';
                videoFeed.removeAttribute('src');
            }
            if (placeholder) placeholder.style.display = 'flex';
            
            // Reset counts
            document.getElementById('peopleCount').textContent = '0';
            document.getElementById('fpsCount').textContent = '0';
        }
    } catch (error) {
        console.error('Error stopping monitoring:', error);
    }
}

async function toggleHeatmap() {
    const heatmapBtn = document.getElementById('heatmapBtn');
    
    try {
        const response = await fetch('/api/toggle_heatmap', { method: 'POST' });
        const data = await response.json();
        
        if (data.heatmap_enabled) {
            heatmapBtn.classList.add('active');
        } else {
            heatmapBtn.classList.remove('active');
        }
    } catch (error) {
        console.error('Error toggling heatmap:', error);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (dashboard && dashboard.isRunning) {
        // Stop monitoring before page closes
        await fetch('/api/stop', { method: 'POST' });
    }
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new ZaytricsDashboard();
    
    // Add scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for scroll animations
    document.querySelectorAll('.feature-card, .dashboard-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});
