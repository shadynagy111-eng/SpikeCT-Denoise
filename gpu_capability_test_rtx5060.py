"""
GPU Capability Test for RTX 5060 Laptop - Mayo Chest Dataset
Tests memory limits, batch sizes, and training configurations

Hardware: RTX 5060 Laptop GPU (8GB VRAM)
Dataset: Mayo Chest CT (50 patients, 512x512)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from datetime import datetime

class TestCNN(nn.Module):
    """Simplified CNN baseline for memory testing"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TestSNN(nn.Module):
    """Simplified SNN for memory testing (with temporal dimension)"""
    def __init__(self, T=30):
        super().__init__()
        self.T = T
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        # x: [B, T, H, W] - temporal input
        B, T, H, W = x.shape
        outputs = []
        
        for t in range(T):
            xt = x[:, t:t+1, :, :]  # [B, 1, H, W]
            xt = self.encoder(xt)
            xt = self.decoder(xt)
            outputs.append(xt)
        
        return torch.stack(outputs, dim=1)  # [B, T, 1, H, W]


def test_gpu_info():
    """Print GPU information"""
    print("="*70)
    print("GPU HARDWARE INFORMATION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Check PyTorch installation.")
        return False
    
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    
    # Memory info
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU Memory: {total_mem:.2f} GB")
    print("="*70)
    return True


def test_memory_limits(model_type='cnn', batch_size=1, T=30, resolution=512):
    """
    Test memory consumption for given configuration
    
    Args:
        model_type: 'cnn' or 'snn'
        batch_size: Number of images per batch
        T: Timesteps (for SNN only)
        resolution: Image resolution (512 or 256)
    """
    device = torch.device('cuda')
    
    print(f"\n{'='*70}")
    print(f"TEST: {model_type.upper()} | Batch={batch_size} | T={T if model_type=='snn' else 'N/A'} | Res={resolution}")
    print(f"{'='*70}")
    
    try:
        # Create model
        if model_type == 'cnn':
            model = TestCNN().to(device)
            x = torch.randn(batch_size, 1, resolution, resolution).to(device)
        else:  # snn
            model = TestSNN(T=T).to(device)
            x = torch.randn(batch_size, T, resolution, resolution).to(device)
        
        target = torch.randn_like(x[:, 0:1] if model_type == 'snn' else x).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        start_time = time.time()
        output = model(x)
        if model_type == 'snn':
            output = output.mean(dim=1)  # Average over time
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        forward_backward_time = time.time() - start_time
        
        # Memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ SUCCESS")
        print(f"Memory Allocated: {allocated:.3f} GB")
        print(f"Memory Reserved: {reserved:.3f} GB")
        print(f"Peak Memory: {peak:.3f} GB")
        print(f"Forward+Backward Time: {forward_backward_time:.3f}s")
        
        # Cleanup
        del model, x, target, output, loss
        torch.cuda.empty_cache()
        
        return True, peak
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ OUT OF MEMORY")
            torch.cuda.empty_cache()
            return False, 0
        else:
            raise e


def run_comprehensive_test():
    """Run comprehensive capability tests for RTX 5060"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE GPU CAPABILITY TEST - RTX 5060 LAPTOP")
    print("Dataset: Mayo Chest CT (50 patients, 512x512)")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    if not test_gpu_info():
        return
    
    results = []
    
    # TEST 1: CNN Baseline (should work easily with 8GB VRAM)
    print("\n" + "🔷"*35)
    print("TEST 1: CNN BASELINE CAPACITY")
    print("🔷"*35)
    
    for batch_size in [1, 2, 4, 8, 16]:
        success, peak_mem = test_memory_limits('cnn', batch_size=batch_size, resolution=512)
        results.append({
            'model': 'CNN',
            'batch': batch_size,
            'T': None,
            'resolution': 512,
            'success': success,
            'peak_mem': peak_mem
        })
        if not success:
            print(f"⚠️ Max CNN batch size: {batch_size//2}")
            break
    
    # TEST 2: SNN with T=30 at full resolution (512x512)
    print("\n" + "🔷"*35)
    print("TEST 2: SNN WITH T=30 (FULL RESOLUTION 512x512)")
    print("🔷"*35)
    
    for batch_size in [1, 2, 4]:
        success, peak_mem = test_memory_limits('snn', batch_size=batch_size, T=30, resolution=512)
        results.append({
            'model': 'SNN',
            'batch': batch_size,
            'T': 30,
            'resolution': 512,
            'success': success,
            'peak_mem': peak_mem
        })
        if not success:
            print(f"⚠️ Max SNN batch size (512x512): {batch_size//2 if batch_size > 1 else 1}")
            break
    
    # TEST 3: SNN with different T values (batch=1)
    print("\n" + "🔷"*35)
    print("TEST 3: SNN - VARYING T (BATCH=1, 512x512)")
    print("🔷"*35)
    
    for T in [16, 20, 24, 28, 30, 32]:
        success, peak_mem = test_memory_limits('snn', batch_size=1, T=T, resolution=512)
        results.append({
            'model': 'SNN',
            'batch': 1,
            'T': T,
            'resolution': 512,
            'success': success,
            'peak_mem': peak_mem
        })
        if not success:
            print(f"⚠️ Max T at batch=1: {T-4}")
            break
    
    # TEST 4: Patch-based training (256x256) as fallback
    print("\n" + "🔷"*35)
    print("TEST 4: PATCH-BASED TRAINING (256x256)")
    print("🔷"*35)
    
    for batch_size in [1, 2, 4, 8]:
        success, peak_mem = test_memory_limits('snn', batch_size=batch_size, T=30, resolution=256)
        results.append({
            'model': 'SNN_PATCH',
            'batch': batch_size,
            'T': 30,
            'resolution': 256,
            'success': success,
            'peak_mem': peak_mem
        })
        if not success:
            print(f"⚠️ Max patch batch size: {batch_size//2}")
            break
    
    # SUMMARY REPORT
    print("\n" + "="*70)
    print("📊 SUMMARY REPORT - RTX 5060 (8GB VRAM)")
    print("="*70)
    
    print("\n✅ SUCCESSFUL CONFIGURATIONS:")
    for r in results:
        if r['success']:
            if r['model'] == 'CNN':
                print(f"  - CNN: batch={r['batch']}, res={r['resolution']}, mem={r['peak_mem']:.2f}GB")
            elif r['model'] == 'SNN':
                print(f"  - SNN: batch={r['batch']}, T={r['T']}, res={r['resolution']}, mem={r['peak_mem']:.2f}GB")
            else:  # SNN_PATCH
                print(f"  - SNN (Patch): batch={r['batch']}, T={r['T']}, res={r['resolution']}, mem={r['peak_mem']:.2f}GB")
    
    print("\n❌ FAILED CONFIGURATIONS:")
    failed = [r for r in results if not r['success']]
    if failed:
        for r in failed:
            if r['model'] == 'CNN':
                print(f"  - CNN: batch={r['batch']}, res={r['resolution']}")
            elif r['model'] == 'SNN':
                print(f"  - SNN: batch={r['batch']}, T={r['T']}, res={r['resolution']}")
            else:
                print(f"  - SNN (Patch): batch={r['batch']}, T={r['T']}, res={r['resolution']}")
    else:
        print("  None - All tests passed! 🎉")
    
    # RECOMMENDATIONS
    print("\n" + "="*70)
    print("💡 TRAINING RECOMMENDATIONS FOR RTX 5060")
    print("="*70)
    
    # Find max successful SNN batch size at T=30, 512x512
    snn_full_res = [r for r in results if r['model'] == 'SNN' and r['T'] == 30 and r['resolution'] == 512 and r['success']]
    
    if snn_full_res:
        max_batch = max([r['batch'] for r in snn_full_res])
        print(f"\n✅ Maximum SNN batch size (T=30, 512x512): {max_batch}")
        
        if max_batch == 1:
            print("\n⚠️  RTX 5060 (8GB) Analysis:")
            print("   - Can only use batch_size=1 for full resolution SNN")
            print("   - RECOMMENDATION: Use gradient accumulation (8 steps)")
            print("   - Effective batch size: 1 × 8 = 8")
            print("   - Training will be slower but achievable")
        elif max_batch == 2:
            print("\n✅ RTX 5060 (8GB) Analysis:")
            print("   - Can use batch_size=2")
            print("   - RECOMMENDATION: Use gradient accumulation (4 steps)")
            print("   - Effective batch size: 2 × 4 = 8")
        else:
            print(f"\n🎉 RTX 5060 (8GB) Analysis:")
            print(f"   - Can use batch_size={max_batch} - Excellent!")
            print(f"   - Gradient accumulation optional")
    else:
        print("\n❌ Cannot fit SNN at 512x512 resolution on RTX 5060")
        print("   - RECOMMENDATION: Use patch-based training (256x256)")
        
        patch_results = [r for r in results if r['model'] == 'SNN_PATCH' and r['success']]
        if patch_results:
            max_patch_batch = max([r['batch'] for r in patch_results])
            print(f"   - Max patch batch size: {max_patch_batch}")
            print(f"   - Extract 4 patches per 512x512 image")
            print(f"   - Test on full 512x512 images")
    
    # Find max T
    snn_t_results = [r for r in results if r['model'] == 'SNN' and r['batch'] == 1 and r['resolution'] == 512 and r['success']]
    if snn_t_results:
        max_t = max([r['T'] for r in snn_t_results])
        print(f"\n📊 Maximum T (timesteps) at batch=1: {max_t}")
        if max_t >= 30:
            print("   ✅ T=30 is achievable (your selected value)")
        else:
            print(f"   ⚠️ T=30 may not fit, consider reducing to T={max_t}")
    
    print("\n" + "="*70)
    print("Test completed:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    print("\nSave this report for your thesis methodology section!")


if __name__ == "__main__":
    run_comprehensive_test()
