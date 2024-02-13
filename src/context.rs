use boojum::cs::implementations::utils::domain_generator_for_size;
use boojum::fft::{bitreverse_enumeration_inplace, distribute_powers};
use boojum::field::goldilocks::GoldilocksField;
use boojum::field::{Field, PrimeField};
use cudart::memory::{memory_copy, DeviceAllocation};
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::DeviceSlice;
use cudart_sys::{cudaGetSymbolAddress, cudaMemcpyToSymbol, CudaMemoryCopyKind};
use std::mem::size_of;
use std::os::raw::c_void;

pub const OMEGA_LOG_ORDER: u32 = 24;

#[repr(C)]
struct PowersLayerData {
    values: *const GoldilocksField,
    mask: u32,
    log_count: u32,
}

impl PowersLayerData {
    fn new(values: *const GoldilocksField, log_count: u32) -> Self {
        let mask = (1 << log_count) - 1;
        Self {
            values,
            mask,
            log_count,
        }
    }
}

#[repr(C)]
struct PowersData {
    fine: PowersLayerData,
    coarse: PowersLayerData,
}

impl PowersData {
    fn new(
        fine_values: *const GoldilocksField,
        fine_log_count: u32,
        coarse_values: *const GoldilocksField,
        coarse_log_count: u32,
    ) -> Self {
        let fine = PowersLayerData::new(fine_values, fine_log_count);
        let coarse = PowersLayerData::new(coarse_values, coarse_log_count);
        Self { fine, coarse }
    }
}

const FINEST_LOG_COUNT: usize = 7;
const COARSER_LOG_COUNT: usize = 8;
const COARSEST_LOG_COUNT: usize = 6;

extern "C" {
    static powers_data_w: PowersData;
    static powers_data_w_bitrev_for_ntt: PowersData;
    static powers_data_w_inv_bitrev_for_ntt: PowersData;

    static ntt_w_powers_bitrev_finest: [GoldilocksField; 1 << FINEST_LOG_COUNT];
    static ntt_w_powers_bitrev_coarser: [GoldilocksField; 1 << COARSER_LOG_COUNT];
    static ntt_w_powers_bitrev_coarsest: [GoldilocksField; 1 << COARSEST_LOG_COUNT];
    static ntt_w_inv_powers_bitrev_finest: [GoldilocksField; 1 << FINEST_LOG_COUNT];
    static ntt_w_inv_powers_bitrev_coarser: [GoldilocksField; 1 << COARSER_LOG_COUNT];
    static ntt_w_inv_powers_bitrev_coarsest: [GoldilocksField; 1 << COARSEST_LOG_COUNT];

    static powers_data_g_f: PowersData;
    static powers_data_g_i: PowersData;
    static inv_sizes: [GoldilocksField; OMEGA_LOG_ORDER as usize + 1];
}

unsafe fn copy_to_symbol<T>(symbol: &T, src: &T) -> CudaResult<()> {
    cudaMemcpyToSymbol(
        symbol as *const T as *const c_void,
        src as *const T as *const c_void,
        size_of::<T>(),
        0,
        CudaMemoryCopyKind::HostToDevice,
    )
    .wrap()
}

#[allow(clippy::too_many_arguments)]
unsafe fn copy_to_symbols(
    powers_of_w_coarse_log_count: u32,
    powers_of_w_fine: *const GoldilocksField,
    powers_of_w_coarse: *const GoldilocksField,
    powers_of_w_fine_bitrev_for_ntt: *const GoldilocksField,
    powers_of_w_coarse_bitrev_for_ntt: *const GoldilocksField,
    powers_of_w_inv_fine_bitrev_for_ntt: *const GoldilocksField,
    powers_of_w_inv_coarse_bitrev_for_ntt: *const GoldilocksField,
    powers_of_g_coarse_log_count: u32,
    powers_of_g_f_fine: *const GoldilocksField,
    powers_of_g_f_coarse: *const GoldilocksField,
    powers_of_g_i_fine: *const GoldilocksField,
    powers_of_g_i_coarse: *const GoldilocksField,
    inv_sizes_host: [GoldilocksField; OMEGA_LOG_ORDER as usize + 1],
) -> CudaResult<()> {
    let coarse_log_count = powers_of_w_coarse_log_count;
    let fine_log_count = OMEGA_LOG_ORDER - coarse_log_count;
    copy_to_symbol(
        &powers_data_w,
        &PowersData::new(
            powers_of_w_fine,
            fine_log_count,
            powers_of_w_coarse,
            coarse_log_count,
        ),
    )?;
    // Accounts for twiddle arrays only covering half the range
    let fine_log_count = fine_log_count - 1;
    copy_to_symbol(
        &powers_data_w_bitrev_for_ntt,
        &PowersData::new(
            powers_of_w_fine_bitrev_for_ntt,
            fine_log_count,
            powers_of_w_coarse_bitrev_for_ntt,
            coarse_log_count,
        ),
    )?;
    copy_to_symbol(
        &powers_data_w_inv_bitrev_for_ntt,
        &PowersData::new(
            powers_of_w_inv_fine_bitrev_for_ntt,
            fine_log_count,
            powers_of_w_inv_coarse_bitrev_for_ntt,
            coarse_log_count,
        ),
    )?;
    let coarse_log_count = powers_of_g_coarse_log_count;
    let fine_log_count = OMEGA_LOG_ORDER - coarse_log_count;
    copy_to_symbol(
        &powers_data_g_f,
        &PowersData::new(
            powers_of_g_f_fine,
            fine_log_count,
            powers_of_g_f_coarse,
            coarse_log_count,
        ),
    )?;
    copy_to_symbol(
        &powers_data_g_i,
        &PowersData::new(
            powers_of_g_i_fine,
            coarse_log_count,
            powers_of_g_i_coarse,
            fine_log_count,
        ),
    )?;
    copy_to_symbol(&inv_sizes, &inv_sizes_host)?;
    Ok(())
}

fn generate_powers_dev(
    base: GoldilocksField,
    powers_dev: &mut DeviceSlice<GoldilocksField>,
    bit_reverse: bool,
) -> CudaResult<()> {
    let mut powers_host = vec![GoldilocksField::ONE; powers_dev.len()];
    distribute_powers(&mut powers_host, base);
    if bit_reverse {
        bitreverse_enumeration_inplace(&mut powers_host);
    }
    memory_copy(powers_dev, &powers_host)
}

fn populate_ntt_powers() -> CudaResult<()> {
    let mut powers_host = [GoldilocksField::ONE; 1 << FINEST_LOG_COUNT];
    let base = domain_generator_for_size::<GoldilocksField>(1u64 << OMEGA_LOG_ORDER);
    // distribute_powers(&mut powers_host, base);
    bitreverse_enumeration_inplace(&mut powers_host);
    unsafe { copy_to_symbol(&ntt_w_powers_bitrev_finest, &powers_host)? };
    let base = base.inverse().expect("must exist");
    // distribute_powers(&mut powers_host, base);
    bitreverse_enumeration_inplace(&mut powers_host);
    unsafe { copy_to_symbol(&ntt_w_inv_powers_bitrev_finest, &powers_host)? };

    let mut powers_host = [GoldilocksField::ONE; 1 << COARSER_LOG_COUNT];
    let base = domain_generator_for_size::<GoldilocksField>(1u64 << (OMEGA_LOG_ORDER - FINEST_LOG_COUNT as u32));
    // distribute_powers(&mut powers_host, base);
    bitreverse_enumeration_inplace(&mut powers_host);
    unsafe { copy_to_symbol(&ntt_w_powers_bitrev_coarser, &powers_host)? };
    let base = base.inverse().expect("must exist");
    // distribute_powers(&mut powers_host, base);
    bitreverse_enumeration_inplace(&mut powers_host);
    unsafe { copy_to_symbol(&ntt_w_inv_powers_bitrev_coarser, &powers_host)? };

    let mut powers_host = [GoldilocksField::ONE; 1 << COARSEST_LOG_COUNT];
    let base = domain_generator_for_size::<GoldilocksField>(1u64 << (COARSEST_LOG_COUNT + 1));
    // distribute_powers(&mut powers_host, base);
    bitreverse_enumeration_inplace(&mut powers_host);
    unsafe { copy_to_symbol(&ntt_w_powers_bitrev_coarsest, &powers_host)? };
    let base = base.inverse().expect("must exist");
    // distribute_powers(&mut powers_host, base);
    bitreverse_enumeration_inplace(&mut powers_host);
    unsafe { copy_to_symbol(&ntt_w_inv_powers_bitrev_coarsest, &powers_host)? };

    Ok(())
}

pub struct Context {
    pub powers_of_w_fine: DeviceAllocation<GoldilocksField>,
    pub powers_of_w_coarse: DeviceAllocation<GoldilocksField>,
    pub powers_of_w_fine_bitrev_for_ntt: DeviceAllocation<GoldilocksField>,
    pub powers_of_w_coarse_bitrev_for_ntt: DeviceAllocation<GoldilocksField>,
    pub powers_of_w_inv_fine_bitrev_for_ntt: DeviceAllocation<GoldilocksField>,
    pub powers_of_w_inv_coarse_bitrev_for_ntt: DeviceAllocation<GoldilocksField>,
    pub powers_of_g_f_fine: DeviceAllocation<GoldilocksField>,
    pub powers_of_g_f_coarse: DeviceAllocation<GoldilocksField>,
    pub powers_of_g_i_fine: DeviceAllocation<GoldilocksField>,
    pub powers_of_g_i_coarse: DeviceAllocation<GoldilocksField>,
}

impl Context {
    pub fn create(
        powers_of_w_coarse_log_count: u32,
        powers_of_g_coarse_log_count: u32,
    ) -> CudaResult<Self> {
        assert!(powers_of_w_coarse_log_count <= OMEGA_LOG_ORDER);
        assert!(powers_of_g_coarse_log_count <= OMEGA_LOG_ORDER);

        populate_ntt_powers()?;

        let length_fine = 1usize << (OMEGA_LOG_ORDER - powers_of_w_coarse_log_count);
        let length_coarse = 1usize << powers_of_w_coarse_log_count;
        let mut powers_of_w_fine = DeviceAllocation::<GoldilocksField>::alloc(length_fine)?;
        let mut powers_of_w_coarse = DeviceAllocation::<GoldilocksField>::alloc(length_coarse)?;
        generate_powers_dev(
            domain_generator_for_size::<GoldilocksField>(1u64 << OMEGA_LOG_ORDER),
            &mut powers_of_w_fine,
            false,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<GoldilocksField>(length_coarse as u64),
            &mut powers_of_w_coarse,
            false,
        )?;
        let length_fine = 1usize << (OMEGA_LOG_ORDER - powers_of_w_coarse_log_count - 1);
        let length_coarse = 1usize << powers_of_w_coarse_log_count;
        let mut powers_of_w_fine_bitrev_for_ntt =
            DeviceAllocation::<GoldilocksField>::alloc(length_fine)?;
        let mut powers_of_w_coarse_bitrev_for_ntt =
            DeviceAllocation::<GoldilocksField>::alloc(length_coarse)?;
        let mut powers_of_w_inv_fine_bitrev_for_ntt =
            DeviceAllocation::<GoldilocksField>::alloc(length_fine)?;
        let mut powers_of_w_inv_coarse_bitrev_for_ntt =
            DeviceAllocation::<GoldilocksField>::alloc(length_coarse)?;
        generate_powers_dev(
            domain_generator_for_size::<GoldilocksField>(1u64 << OMEGA_LOG_ORDER),
            &mut powers_of_w_fine_bitrev_for_ntt,
            true,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<GoldilocksField>((length_coarse * 2) as u64),
            &mut powers_of_w_coarse_bitrev_for_ntt,
            true,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<GoldilocksField>(1u64 << OMEGA_LOG_ORDER)
                .inverse()
                .expect("must exist"),
            &mut powers_of_w_inv_fine_bitrev_for_ntt,
            true,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<GoldilocksField>((length_coarse * 2) as u64)
                .inverse()
                .expect("must exist"),
            &mut powers_of_w_inv_coarse_bitrev_for_ntt,
            true,
        )?;
        let length_fine = 1usize << (OMEGA_LOG_ORDER - powers_of_g_coarse_log_count);
        let length_coarse = 1usize << powers_of_g_coarse_log_count;
        let mut powers_of_g_f_fine = DeviceAllocation::<GoldilocksField>::alloc(length_fine)?;
        let mut powers_of_g_f_coarse = DeviceAllocation::<GoldilocksField>::alloc(length_coarse)?;
        let mut powers_of_g_i_fine = DeviceAllocation::<GoldilocksField>::alloc(length_fine)?;
        let mut powers_of_g_i_coarse = DeviceAllocation::<GoldilocksField>::alloc(length_coarse)?;
        generate_powers_dev(
            GoldilocksField::multiplicative_generator(),
            &mut powers_of_g_f_fine,
            false,
        )?;
        generate_powers_dev(
            GoldilocksField::multiplicative_generator().pow_u64(length_fine as u64),
            &mut powers_of_g_f_coarse,
            false,
        )?;
        let g_inv = GoldilocksField::multiplicative_generator()
            .inverse()
            .expect("inv of generator must exist");
        generate_powers_dev(g_inv, &mut powers_of_g_i_fine, false)?;
        generate_powers_dev(
            g_inv.pow_u64(length_fine as u64),
            &mut powers_of_g_i_coarse,
            false,
        )?;
        let two_inv = GoldilocksField(2).inverse().expect("must exist");
        let mut inv_sizes_host = [GoldilocksField::ONE; (OMEGA_LOG_ORDER + 1) as usize];
        distribute_powers(&mut inv_sizes_host, two_inv);
        unsafe {
            copy_to_symbols(
                powers_of_w_coarse_log_count,
                powers_of_w_fine.as_ptr(),
                powers_of_w_coarse.as_ptr(),
                powers_of_w_fine_bitrev_for_ntt.as_ptr(),
                powers_of_w_coarse_bitrev_for_ntt.as_ptr(),
                powers_of_w_inv_fine_bitrev_for_ntt.as_ptr(),
                powers_of_w_inv_coarse_bitrev_for_ntt.as_ptr(),
                powers_of_g_coarse_log_count,
                powers_of_g_f_fine.as_ptr(),
                powers_of_g_f_coarse.as_ptr(),
                powers_of_g_i_fine.as_ptr(),
                powers_of_g_i_coarse.as_ptr(),
                inv_sizes_host,
            )?;
        }
        Ok(Self {
            powers_of_w_fine,
            powers_of_w_coarse,
            powers_of_w_fine_bitrev_for_ntt,
            powers_of_w_coarse_bitrev_for_ntt,
            powers_of_w_inv_fine_bitrev_for_ntt,
            powers_of_w_inv_coarse_bitrev_for_ntt,
            powers_of_g_f_fine,
            powers_of_g_f_coarse,
            powers_of_g_i_fine,
            powers_of_g_i_coarse,
        })
    }

    pub fn destroy(self) -> CudaResult<()> {
        self.powers_of_w_fine.free()?;
        self.powers_of_w_coarse.free()?;
        self.powers_of_w_fine_bitrev_for_ntt.free()?;
        self.powers_of_w_coarse_bitrev_for_ntt.free()?;
        self.powers_of_w_inv_fine_bitrev_for_ntt.free()?;
        self.powers_of_w_inv_coarse_bitrev_for_ntt.free()?;
        self.powers_of_g_f_fine.free()?;
        self.powers_of_g_f_coarse.free()?;
        self.powers_of_g_i_fine.free()?;
        self.powers_of_g_i_coarse.free()?;
        Ok(())
    }
}
