use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult};
use burn_automata_kernels::{AdaptiveGraphPolicy, AdaptivePerceptionConfig};

/// Maximum number of graded material-slot exchanges performed by one
/// device-resident topology pass.
///
/// The WGPU shader keeps both candidate sets in workgroup memory. Sixty-four
/// amortizes detail capture and submission overhead while remaining a small
/// fixed allocation relative to the 256-lane topology workgroup.
pub const CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveProxyConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_proxy_branch_factor")]
    pub branch_factor: usize,
    /// One-based cache row: 1 groups leaves, 2 groups row 1, and so on.
    #[serde(default = "default_proxy_level")]
    pub level: usize,
    #[serde(default = "default_proxy_context_scale")]
    pub context_scale: f32,
    #[serde(default = "default_proxy_bandwidth_scale")]
    pub bandwidth_scale: f32,
}

impl Default for AdaptiveProxyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            branch_factor: default_proxy_branch_factor(),
            level: default_proxy_level(),
            context_scale: default_proxy_context_scale(),
            bandwidth_scale: default_proxy_bandwidth_scale(),
        }
    }
}

const fn default_proxy_branch_factor() -> usize {
    4
}

const fn default_proxy_level() -> usize {
    2
}

const fn default_proxy_context_scale() -> f32 {
    1.0
}

const fn default_proxy_bandwidth_scale() -> f32 {
    2.5
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveRulePerception {
    /// Represented-measure SPH with an exact fixed-NPA limit. Rollouts keep the
    /// source rule's bandwidth fixed because pretrained NPA rules were not
    /// optimized for changing support radii.
    #[default]
    NpaCompatible,
    /// Normalized Shepard and moment-corrected paper operator. Rules using this
    /// mode must be trained against these feature semantics.
    NormalizedAdaptive,
}

/// Initial distribution of conserved material measure over a fixed active-row
/// budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveMaterialSeedLayout {
    /// Restrict a fine seed through conservative hierarchy groups. This is the
    /// canonical hard-event layout and may contain dyadic coarse rows.
    #[default]
    CanonicalGrouped,
    /// Distribute total measure uniformly over every active row. The resulting
    /// noninteger material scale is a continuous fixed-budget control; hard
    /// canonical topology is intentionally inapplicable.
    UniformContinuous,
    /// Distribute a deterministic log-graded continuum of represented measure
    /// over every active row. The multiset is fixed and conserves the same
    /// reference material as the uniform control; runtime reallocation may
    /// move these material slots without introducing hidden rows.
    GradedContinuous,
}

/// Numerical realization used for material leaves larger than the native NPA
/// particle measure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveCoarseDynamics {
    /// Evaluate the represented-measure SPH operator directly on active leaves.
    #[default]
    RepresentedMeasure,
    /// Evaluate the frozen rule on the conservative bootstrap quadrature held
    /// by each coarse leaf, then restrict its physical update.
    FineQuadrature,
    /// Diagnostic ceiling that evolves every retained quadrature child and
    /// restricts its material moments back to the active coarse leaf. This
    /// preserves fine latent degrees of freedom and is therefore not an
    /// adaptive-compute result.
    PersistentFineQuadrature,
}

/// Target-independent policy used by a scheduled fine-to-coarse hierarchy cut.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveHierarchyRestrictionPolicy {
    /// Merge the most spatially compact hierarchy groups first. This is the
    /// inexpensive, target-independent bootstrap policy for direct active-
    /// material training and does not evaluate the NPA rule during seeding.
    SpatialCompactness,
    /// Rank hierarchy detail from the frozen NPA update and material state.
    #[default]
    DynamicsDetail,
    /// Rank first-level sibling groups with the model's dedicated learned
    /// restriction controller. The target image is never consulted at runtime.
    LearnedController,
}

/// Conservative event family used by the scheduled hierarchy restriction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveRestrictionArity {
    /// Use complete `2d`-child canonical merges. This preserves legacy
    /// artifacts and produces dyadic material radii.
    #[default]
    Canonical,
    /// Use deterministic 2/3/4-child subsets from first-level spatial groups.
    /// Every fine leaf belongs to exactly one material aggregate, so measure,
    /// centroid, covariance, and intensive-state restriction remain exact.
    Mixed,
}

/// How a multi-interval scheduled restriction updates its spatial cut.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveRestrictionSchedule {
    /// Re-rank the full fine state at each interval. This is a dynamic LoD
    /// reallocation policy: aggregate identities may move as detail evolves.
    #[default]
    RollingRecompute,
    /// Re-rank current fine material while replacing at most
    /// `max_events_per_interval` existing canonical groups. Coherent groups
    /// retain identity; the remainder is rebuilt from current controller
    /// diagnostics so the cut can track a moving rollout without a full-frame
    /// topology exchange.
    BoundedRollingRecompute,
    /// Preserve every previously selected aggregate and allocate only from
    /// still-fine material. This provides stable nested LoD identities.
    Nested,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveResidualGateReference {
    /// Preserve legacy artifacts: residual level zero is the frozen rule's
    /// native particle footprint.
    #[default]
    BaseRule,
    /// Residual level zero is the equal-measure deployment budget. This keeps
    /// the fair fixed-budget control untouched while correcting only its
    /// reallocated fine and coarse leaves.
    TargetBudget,
}

/// How the optional normalized-perception local rule is composed with the
/// NPA-compatible base rule at runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveLocalRuleSemantics {
    /// Add a footprint-gated correction to the base rule.
    #[default]
    Residual,
    /// Add a normalized-perception correction to coarse rows and to native
    /// rows whose support contains coarse represented-measure sources.
    NormalizedExposureResidual,
    /// Add a mixed-resolution correction that consumes the same NPA-compatible
    /// perception as the frozen base. The branch is disabled for a uniform
    /// native-scale material system and active for every row once coarse
    /// material is present, because fine targets also observe coarse sources.
    /// This preserves the fixed-NPA limit and avoids a second traversal.
    CompatibleResidual,
    /// Preserve the base rule exactly for native leaves and replace its update
    /// with the normalized local rule only for leaves above the native
    /// material footprint.
    CoarseReplacement,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveTopologyControl {
    #[default]
    Learned,
    /// Use the deterministic one-level refinement defect to rank resolution
    /// changes while retaining learned global split/merge gates and per-leaf
    /// bandwidth modulation. This avoids checkpoint-sensitive candidate
    /// ordering without bypassing the trained controller's topology gate.
    LearnedRefinementDefect,
    LocalDetailOracle,
    /// Reallocate a fixed material budget with the same detached local-detail
    /// rule used by adaptive Target2D training: merge four compact fine rows
    /// into one coarse slot and split one high-detail coarse row into the four
    /// vacated fine slots. This path does not depend on the learned topology
    /// controller.
    PairedLocalDetail,
    /// Reallocate a fixed continuum of material scales without changing row
    /// count: exchange the finest low-detail slot with the coarsest high-detail
    /// slot, then correct the intensive fields to conserve their represented
    /// material moments. This is the deployment policy for
    /// `graded-continuous` material seeds.
    ContinuousLocalDetail,
    RefinementDefectOracle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveNpaConfig {
    pub spatial_dims: usize,
    pub controller_hidden_dims: usize,
    pub reference_footprint: f32,
    /// Material footprint at which the frozen base NPA is authoritative.
    /// Zero in older artifacts falls back to `reference_footprint`.
    #[serde(default)]
    pub base_rule_footprint: f32,
    /// Scale of the optional learned leaf-local correction. Residual semantics
    /// use it as a correction gain; coarse-replacement semantics use a bounded
    /// interpolation from the frozen base (zero) to the learned rule (one).
    #[serde(default = "default_local_residual_scale")]
    pub local_residual_scale: f32,
    /// Output-group calibration for learned position updates. Coarse
    /// replacement keeps this at one so its interpolation remains convex.
    #[serde(default = "default_local_residual_scale")]
    pub local_residual_motion_scale: f32,
    /// Output-group calibration for learned latent-state updates.
    #[serde(default = "default_local_residual_scale")]
    pub local_residual_state_scale: f32,
    #[serde(default)]
    pub residual_gate_reference: AdaptiveResidualGateReference,
    /// Runtime composition of the optional normalized local rule. This is
    /// serialized independently of the training recipe so inference cannot
    /// reinterpret a trained branch.
    #[serde(default)]
    pub local_rule_semantics: AdaptiveLocalRuleSemantics,
    /// Append budget-relative scale and retained affine-detail magnitude to
    /// the leaf-local residual input. The two channels occupy the local rule's
    /// position-feature slots; the frozen base rule remains unchanged.
    #[serde(default)]
    pub closure_moment_features: bool,
    /// Append one compact affine-null recurrent state value per NPA state
    /// channel to the coarse closure branch. This is opt-in so existing
    /// artifacts retain their exact input layout.
    #[serde(default)]
    pub closure_recurrent_mode: bool,
    /// Recurrent channels inserted immediately before the RGB tail of the
    /// direct active-material NPA state. They carry compact unresolved
    /// sub-leaf memory without adding hidden particles or rendered capacity.
    #[serde(default)]
    pub compact_recurrent_memory_dims: usize,
    /// Append one continuous material-footprint-relative scale channel to the
    /// primary shared NPA rule. The native fine footprint maps to zero, so a
    /// zero-initialized input column preserves the source rule exactly while
    /// allowing one rule to specialize across material scales independently of
    /// the communication-support policy.
    #[serde(default)]
    pub material_scale_conditioning: bool,
    /// Serialized seed-layout contract used by training, validation, and
    /// inference. Older artifacts retain canonical grouped seeding.
    #[serde(default)]
    pub material_seed_layout: AdaptiveMaterialSeedLayout,
    /// Exponent mapping the continuous represented-measure ratio to the seed
    /// interaction bandwidth. This is used only by continuous seeds.
    #[serde(default)]
    pub material_seed_bandwidth_exponent: f32,
    /// Largest-to-smallest represented-measure ratio in a graded continuous
    /// seed. One preserves the uniform continuous control.
    #[serde(default = "default_material_seed_measure_ratio")]
    pub material_seed_measure_ratio: f32,
    /// Give a compatible frozen-base residual two explicit channels: relative
    /// material scale and the local fraction of density contributed by coarse
    /// sources. This keeps unaffected fine rows exactly on the validated base
    /// rule while making the mixed-resolution closure identifiable.
    #[serde(default)]
    pub compatible_residual_material_features: bool,
    /// Smallest material footprint eligible for topology and dynamics.
    pub min_footprint: f32,
    /// Largest material footprint eligible for topology and dynamics.
    pub max_footprint: f32,
    /// Smallest Gaussian footprint used for visualization. Zero preserves
    /// legacy behavior and reuses `min_footprint`.
    #[serde(default)]
    pub min_render_footprint: f32,
    /// Largest Gaussian footprint used for visualization. Zero preserves
    /// legacy behavior and reuses `max_footprint`.
    #[serde(default)]
    pub max_render_footprint: f32,
    /// Exponent applied around `reference_footprint` before display clamping.
    /// Values above one emphasize coarse/fine scale contrast without changing
    /// represented measure, perception, or topology.
    #[serde(default = "default_render_footprint_exponent")]
    pub render_footprint_exponent: f32,
    pub min_leaves: usize,
    pub max_leaves: usize,
    /// Steady-state visible material budget after optional bootstrap and
    /// scheduled restriction.
    pub target_leaves: usize,
    /// Temporary coarse-to-fine fill budget. Zero preserves legacy behavior
    /// and fills directly to `target_leaves`. A larger value allows an LoD
    /// seed to recover the fine NPA trajectory before a scheduled restriction
    /// selects the steady mixed-resolution cut.
    #[serde(default)]
    pub bootstrap_target_leaves: usize,
    /// Material leaves used at rollout initialization. Zero preserves legacy
    /// behavior and starts directly at `target_leaves`.
    #[serde(default)]
    pub initial_leaves: usize,
    /// Build a coarse bootstrap seed by conservatively restricting a matched
    /// `target_leaves` seed. This preserves the target seed's material moments
    /// and avoids clustered children from an independently sampled coarse
    /// population.
    #[serde(default = "default_true")]
    pub hierarchical_bootstrap_seed: bool,
    /// Fine reference population conservatively restricted by the hierarchical
    /// bootstrap. Zero preserves the legacy behavior and uses `target_leaves`.
    /// A larger value permits a target budget to be a nonuniform cut of one
    /// shared fine seed instead of an equal-measure population at that budget.
    #[serde(default)]
    pub bootstrap_fine_leaves: usize,
    /// Optional absolute rollout step after which a fine initial population is
    /// conservatively restricted to `target_leaves` using its current detail.
    /// This is an oracle-allocation control for separating cut quality from
    /// compressed coarse dynamics; zero disables it.
    #[serde(default)]
    pub hierarchical_restriction_step: usize,
    /// Maximum visible-leaf reduction applied per topology interval after
    /// `hierarchical_restriction_step`. Zero preserves the legacy one-shot
    /// restriction. A bounded value grades a fine-to-coarse LoD transition
    /// while still converging exactly to `target_leaves`.
    #[serde(default)]
    pub hierarchical_restriction_leaf_delta_per_interval: usize,
    /// Event arity used by the scheduled fine-to-coarse restriction.
    #[serde(default)]
    pub hierarchical_restriction_arity: AdaptiveRestrictionArity,
    /// Identity policy used across a progressive restriction schedule.
    #[serde(default)]
    pub hierarchical_restriction_schedule: AdaptiveRestrictionSchedule,
    /// Target-independent policy used by `hierarchical_restriction_step`.
    #[serde(default)]
    pub hierarchical_restriction_policy: AdaptiveHierarchyRestrictionPolicy,
    /// Retain the original fine children behind hierarchy-restricted leaves.
    /// This is useful for diagnostic fine-quadrature ceilings, but it is a
    /// hidden fine-state path and must be disabled for genuine adaptive
    /// recurrent/render deployments.
    #[serde(default = "default_true")]
    pub retain_bootstrap_templates: bool,
    pub topology_interval: usize,
    /// Topology cadence after coarse-to-fine bootstrap reaches its target.
    /// Zero preserves legacy behavior and reuses `topology_interval`.
    #[serde(default)]
    pub steady_topology_interval: usize,
    /// Absolute rollout step before discrete split/merge events are enabled.
    /// This lets morphogenesis establish a meaningful state field before
    /// material resolution is reallocated.
    #[serde(default)]
    pub topology_start_step: usize,
    /// Last absolute rollout step on which topology may change. Zero leaves
    /// topology enabled for the full rollout. This supports bounded adaptation
    /// experiments without encoding one-shot behavior through cadence quirks.
    #[serde(default)]
    pub topology_end_step: usize,
    /// Absolute rollout step before steady budget-neutral split/merge events
    /// are enabled. Zero preserves legacy behavior and reuses
    /// `topology_start_step`; coarse-to-fine bootstrap still uses
    /// `topology_start_step` so an unformed seed can be refined immediately.
    #[serde(default)]
    pub steady_topology_start_step: usize,
    /// Topology policy embedded in the model artifact and used by the public
    /// CPU and WGPU inference entrypoints. Older artifacts retain their learned
    /// controller-only behavior when this field is absent.
    #[serde(default = "legacy_runtime_topology_control")]
    pub runtime_topology_control: AdaptiveTopologyControl,
    pub max_events_per_interval: usize,
    /// Canonical four-child split-radius multiplier used by
    /// `paired-local-detail` deployment topology.
    #[serde(default = "default_paired_topology_split_radius_scale")]
    pub paired_topology_split_radius_scale: f32,
    /// Local-detail contribution to the compact fine-cluster merge score used
    /// by `paired-local-detail` deployment topology.
    #[serde(default = "default_paired_topology_merge_detail_scale")]
    pub paired_topology_merge_detail_scale: f32,
    /// Required relative improvement for a budget-neutral merge/split pair.
    /// This guards hard event selection against tiny device-reduction changes
    /// near an equal-cost boundary. Zero preserves legacy artifacts; one
    /// disables budget-neutral reallocation while retaining count-changing
    /// bootstrap and restriction events.
    #[serde(default)]
    pub min_reallocation_relative_gain: f32,
    /// Last absolute step of forced coarse-to-fine budget filling. Zero
    /// disables bootstrap and retains the embedded runtime topology policy.
    #[serde(default)]
    pub bootstrap_end_step: usize,
    /// Split budget during coarse-to-fine bootstrap. Zero reuses
    /// `max_events_per_interval`.
    #[serde(default)]
    pub bootstrap_events_per_interval: usize,
    /// World-space child offset used only while refining the unformed seed.
    /// Zero retains the conservative material-scale canonical split. A
    /// positive value decorrelates bootstrap siblings before recurrent
    /// dynamics; steady material topology always uses the canonical split.
    #[serde(default)]
    pub bootstrap_seed_spread: f32,
    /// Log-scale Gaussian transition duration after topology changes.
    #[serde(default = "default_render_transition_steps")]
    pub render_transition_steps: u32,
    /// Per-dynamics-step geometric interpolation from inherited display scale
    /// toward the leaf's represented-measure footprint on the CPU path. The
    /// resident GPU path uses `render_transition_steps` for the same purpose.
    #[serde(default = "default_render_footprint_relaxation")]
    pub render_footprint_relaxation: f32,
    /// Smallest desired/current footprint ratio accepted in one topology pass.
    #[serde(default = "default_min_topology_footprint_ratio")]
    pub min_topology_footprint_ratio: f32,
    /// Largest desired/current footprint ratio accepted in one topology pass.
    #[serde(default = "default_max_topology_footprint_ratio")]
    pub max_topology_footprint_ratio: f32,
    pub split_ratio: f32,
    pub merge_ratio: f32,
    /// Maximum represented-measure ratio between children of one accepted
    /// split. One preserves the legacy equal-measure canonical event; values
    /// above one enable conservative continuously weighted children inferred
    /// from the local desired-resolution field.
    #[serde(default = "default_max_unequal_split_measure_ratio")]
    pub max_unequal_split_measure_ratio: f32,
    /// Number of nearby material leaves used to reconstruct the local desired
    /// log-footprint gradient for an unequal split.
    #[serde(default = "default_split_field_neighbors")]
    pub split_field_neighbors: usize,
    /// Maximum footprint ratio across an interacting material pair. Zero
    /// preserves legacy artifacts and disables the explicit grading gate.
    #[serde(default)]
    pub max_neighbor_footprint_ratio: f32,
    pub split_probability: f32,
    pub merge_probability: f32,
    #[serde(default = "default_merge_extent_ratio")]
    pub merge_extent_ratio: f32,
    /// Permit spatial hierarchy buckets to form merge candidates when no
    /// canonical split sibling relationship exists. Disabling this is the
    /// conservative path for recurrent latent states: only exact sibling
    /// groups produced by a prior split may be restricted back to a parent.
    #[serde(default = "default_true")]
    pub spatial_merge_groups_enabled: bool,
    #[serde(default = "default_merge_state_rms_limit")]
    pub merge_state_rms_limit: f32,
    /// Maximum RMS state detail introduced by affine split prolongation. Zero
    /// preserves artifact compatibility and reuses `merge_state_rms_limit`.
    /// Keeping this independent lets merges remain conservative without
    /// suppressing useful child detail at a refinement event.
    #[serde(default)]
    pub split_state_transfer_rms_limit: f32,
    /// Blend applied to the measured affine state-Jacobian prolongation when
    /// splitting a material leaf. Zero copies intensive latent state exactly;
    /// one applies the full affine reconstruction.
    #[serde(default)]
    pub split_state_prolongation_scale: f32,
    pub cooldown_steps: u16,
    pub bandwidth_relaxation: f32,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    #[serde(default)]
    pub rule_perception: AdaptiveRulePerception,
    #[serde(default)]
    pub coarse_dynamics: AdaptiveCoarseDynamics,
    /// Maximum persistent quadrature modes evaluated inside each coarse leaf.
    /// Zero retains every fine child and is an exact diagnostic ceiling. Values
    /// below the canonical child count compress unresolved dynamics while the
    /// active material-leaf budget remains unchanged.
    #[serde(default)]
    pub coarse_quadrature_points: usize,
    /// Persistent modes retained while a coarse-to-fine bootstrap still owns
    /// inactive fine templates. Zero inherits `coarse_quadrature_points`.
    #[serde(default)]
    pub bootstrap_quadrature_points: usize,
    /// Evolve coarse covariance and retained affine state detail under the
    /// spatial Jacobian of the physical NPA update. Native leaves retain the
    /// fixed-resolution integration path exactly.
    #[serde(default)]
    pub transport_coarse_moments: bool,
    /// A coarse leaf represents multiple independently masked fine particles.
    /// Use the update probability as its mean update gate instead of applying
    /// one Bernoulli draw to all represented material.
    #[serde(default)]
    pub expected_coarse_update_mask: bool,
    #[serde(default = "default_rule_graph_policy")]
    pub rule_graph_policy: AdaptiveGraphPolicy,
    #[serde(default)]
    pub proxy: AdaptiveProxyConfig,
    pub perception: AdaptivePerceptionConfig,
}

impl AdaptiveNpaConfig {
    pub fn growing_2d() -> Self {
        Self {
            spatial_dims: 2,
            controller_hidden_dims: 64,
            reference_footprint: 0.025,
            base_rule_footprint: 0.0,
            local_residual_scale: default_local_residual_scale(),
            local_residual_motion_scale: default_local_residual_scale(),
            local_residual_state_scale: default_local_residual_scale(),
            residual_gate_reference: AdaptiveResidualGateReference::default(),
            local_rule_semantics: AdaptiveLocalRuleSemantics::default(),
            closure_moment_features: false,
            closure_recurrent_mode: false,
            compact_recurrent_memory_dims: 0,
            material_scale_conditioning: false,
            material_seed_layout: AdaptiveMaterialSeedLayout::default(),
            material_seed_bandwidth_exponent: 0.0,
            material_seed_measure_ratio: default_material_seed_measure_ratio(),
            compatible_residual_material_features: false,
            min_footprint: 0.0015625,
            max_footprint: 0.1,
            min_render_footprint: 0.0,
            max_render_footprint: 0.0,
            render_footprint_exponent: default_render_footprint_exponent(),
            min_leaves: 256,
            max_leaves: 16_384,
            target_leaves: 4_096,
            bootstrap_target_leaves: 0,
            initial_leaves: 0,
            hierarchical_bootstrap_seed: true,
            bootstrap_fine_leaves: 0,
            hierarchical_restriction_step: 0,
            hierarchical_restriction_leaf_delta_per_interval: 0,
            hierarchical_restriction_arity: AdaptiveRestrictionArity::default(),
            hierarchical_restriction_schedule: AdaptiveRestrictionSchedule::default(),
            hierarchical_restriction_policy: AdaptiveHierarchyRestrictionPolicy::default(),
            retain_bootstrap_templates: true,
            topology_interval: 8,
            steady_topology_interval: 0,
            topology_start_step: 0,
            topology_end_step: 0,
            steady_topology_start_step: 0,
            runtime_topology_control: AdaptiveTopologyControl::LearnedRefinementDefect,
            max_events_per_interval: 64,
            paired_topology_split_radius_scale: default_paired_topology_split_radius_scale(),
            paired_topology_merge_detail_scale: default_paired_topology_merge_detail_scale(),
            min_reallocation_relative_gain: 0.0,
            bootstrap_end_step: 0,
            bootstrap_events_per_interval: 0,
            bootstrap_seed_spread: 0.0,
            render_transition_steps: default_render_transition_steps(),
            render_footprint_relaxation: default_render_footprint_relaxation(),
            min_topology_footprint_ratio: default_min_topology_footprint_ratio(),
            max_topology_footprint_ratio: default_max_topology_footprint_ratio(),
            split_ratio: 0.72,
            merge_ratio: 1.55,
            max_unequal_split_measure_ratio: default_max_unequal_split_measure_ratio(),
            split_field_neighbors: default_split_field_neighbors(),
            max_neighbor_footprint_ratio: 0.0,
            split_probability: 0.55,
            merge_probability: 0.55,
            merge_extent_ratio: default_merge_extent_ratio(),
            spatial_merge_groups_enabled: true,
            merge_state_rms_limit: default_merge_state_rms_limit(),
            split_state_transfer_rms_limit: 0.0,
            split_state_prolongation_scale: 0.0,
            cooldown_steps: 24,
            bandwidth_relaxation: 0.25,
            domain_min: [-1.25, -1.25, 0.0],
            domain_max: [1.25, 1.25, 0.0],
            rule_perception: AdaptiveRulePerception::NpaCompatible,
            coarse_dynamics: AdaptiveCoarseDynamics::default(),
            coarse_quadrature_points: 0,
            bootstrap_quadrature_points: 0,
            transport_coarse_moments: false,
            expected_coarse_update_mask: false,
            rule_graph_policy: default_rule_graph_policy(),
            proxy: AdaptiveProxyConfig::default(),
            perception: AdaptivePerceptionConfig::growing_2d(),
        }
    }

    pub fn sparse_3d() -> Self {
        Self {
            spatial_dims: 3,
            reference_footprint: 0.03,
            min_footprint: 0.0075,
            max_footprint: 0.24,
            min_leaves: 1_024,
            max_leaves: 1_048_576,
            target_leaves: 65_536,
            domain_min: [-1.25; 3],
            domain_max: [1.25; 3],
            perception: AdaptivePerceptionConfig::sparse_3d(),
            ..Self::growing_2d()
        }
    }

    pub fn validate(&self) -> AutomataResult<()> {
        let bootstrap_target = self.bootstrap_target_leaf_count();
        if !(self.spatial_dims == 2 || self.spatial_dims == 3) {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive spatial_dims must be 2 or 3, got {}",
                self.spatial_dims
            )));
        }
        if self.perception.dim != self.spatial_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive perception dim {} does not match model dim {}",
                self.perception.dim, self.spatial_dims
            )));
        }
        if self.closure_recurrent_mode && (!self.closure_moment_features || self.spatial_dims != 2)
        {
            return Err(AutomataError::InvalidArgument(
                "recurrent closure mode currently requires 2D closure-moment features".to_owned(),
            ));
        }
        if self.closure_recurrent_mode
            && self.coarse_dynamics != AdaptiveCoarseDynamics::RepresentedMeasure
        {
            return Err(AutomataError::InvalidArgument(
                "recurrent closure mode requires represented-measure coarse dynamics".to_owned(),
            ));
        }
        if self.compact_recurrent_memory_dims > 8
            || self.spatial_dims != 2 && self.compact_recurrent_memory_dims > 0
        {
            return Err(AutomataError::InvalidArgument(
                "compact recurrent memory currently supports at most eight channels in 2D"
                    .to_owned(),
            ));
        }
        self.perception.validate()?;
        if self.proxy.branch_factor < 2
            || self.proxy.level == 0
            || !self.proxy.context_scale.is_finite()
            || self.proxy.context_scale < 0.0
            || !self.proxy.bandwidth_scale.is_finite()
            || self.proxy.bandwidth_scale <= 0.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive proxy configuration is invalid".to_string(),
            ));
        }
        if self.controller_hidden_dims == 0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive controller_hidden_dims must be non-zero".to_string(),
            ));
        }
        if !self.reference_footprint.is_finite()
            || !self.min_footprint.is_finite()
            || !self.max_footprint.is_finite()
            || self.min_footprint <= 0.0
            || self.reference_footprint < self.min_footprint
            || self.max_footprint < self.reference_footprint
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive footprints must satisfy 0 < min <= reference <= max".to_string(),
            ));
        }
        let min_render_footprint = self.min_render_footprint();
        let max_render_footprint = self.max_render_footprint();
        if !min_render_footprint.is_finite()
            || !max_render_footprint.is_finite()
            || min_render_footprint <= 0.0
            || max_render_footprint < min_render_footprint
            || !self.render_footprint_exponent.is_finite()
            || self.render_footprint_exponent <= 0.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive render footprints require finite 0 < min <= max and exponent > 0"
                    .to_string(),
            ));
        }
        let base_rule_footprint = self.base_rule_footprint();
        if !base_rule_footprint.is_finite()
            || base_rule_footprint < self.min_footprint
            || base_rule_footprint > self.max_footprint
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive base-rule footprint must lie within the material footprint range"
                    .to_string(),
            ));
        }
        if [
            self.local_residual_scale,
            self.local_residual_motion_scale,
            self.local_residual_state_scale,
        ]
        .into_iter()
        .any(|scale| !scale.is_finite() || scale < 0.0)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive local residual scales must be finite and non-negative".to_string(),
            ));
        }
        if self.local_rule_semantics == AdaptiveLocalRuleSemantics::CoarseReplacement
            && (self.rule_perception != AdaptiveRulePerception::NpaCompatible
                || self.residual_gate_reference != AdaptiveResidualGateReference::BaseRule
                || self.local_residual_scale > 1.0
                || (self.local_residual_motion_scale - 1.0).abs() > 1.0e-6
                || (self.local_residual_state_scale - 1.0).abs() > 1.0e-6)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive coarse replacement requires NPA-compatible base perception, base-rule scale classification, a blend in [0, 1], and unit output-group scales"
                    .to_string(),
            ));
        }
        if self.min_leaves == 0
            || self.min_leaves > self.target_leaves
            || self.target_leaves > self.max_leaves
            || bootstrap_target < self.target_leaves
            || bootstrap_target > self.max_leaves
            || self.initial_leaf_count() < self.min_leaves
            || self.initial_leaf_count() > self.max_leaves
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive leaves must satisfy 0 < min <= target <= bootstrap_target <= max with initial in min..=max"
                    .to_string(),
            ));
        }
        if self.bootstrap_end_step > 0 && self.initial_leaf_count() > bootstrap_target {
            return Err(AutomataError::InvalidArgument(
                "adaptive coarse-to-fine bootstrap cannot start above bootstrap_target_leaves"
                    .to_string(),
            ));
        }
        if self.bootstrap_fine_leaf_count() < bootstrap_target {
            return Err(AutomataError::InvalidArgument(
                "adaptive bootstrap fine leaves must be at least bootstrap_target_leaves"
                    .to_string(),
            ));
        }
        if self.coarse_quadrature_points > 2 * self.spatial_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive coarse quadrature points must be zero or at most {}",
                2 * self.spatial_dims,
            )));
        }
        if self.bootstrap_quadrature_points > 2 * self.spatial_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive bootstrap quadrature points must be zero or at most {}",
                2 * self.spatial_dims,
            )));
        }
        let direct_fine_restriction = self.initial_leaf_count() == self.bootstrap_fine_leaf_count()
            && self.initial_leaf_count() > self.target_leaves;
        let progressive_fine_restriction = self.bootstrap_end_step > 0
            && self.initial_leaf_count() < bootstrap_target
            && bootstrap_target == self.bootstrap_fine_leaf_count()
            && bootstrap_target > self.target_leaves
            && self.hierarchical_restriction_step > self.bootstrap_end_step;
        if self.hierarchical_restriction_step > 0
            && (!self.hierarchical_bootstrap_seed
                || !(direct_fine_restriction || progressive_fine_restriction))
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive scheduled restriction requires a hierarchical fine bootstrap target above target_leaves and a restriction step after bootstrap (hierarchical={}, initial={}, target={}, bootstrap_target={}, bootstrap_fine={}, bootstrap_end={}, restriction_step={})",
                self.hierarchical_bootstrap_seed,
                self.initial_leaf_count(),
                self.target_leaves,
                bootstrap_target,
                self.bootstrap_fine_leaf_count(),
                self.bootstrap_end_step,
                self.hierarchical_restriction_step,
            )));
        }
        let event_leaf_delta = 2 * self.spatial_dims - 1;
        if self.hierarchical_restriction_step > 0
            && self.hierarchical_restriction_arity == AdaptiveRestrictionArity::Canonical
            && !(self.bootstrap_fine_leaf_count() - self.target_leaves)
                .is_multiple_of(event_leaf_delta)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive fine/target leaf difference must be divisible by canonical event delta {event_leaf_delta}",
            )));
        }
        if self.hierarchical_bootstrap_seed
            && self.bootstrap_end_step > 0
            && self.initial_leaf_count() < bootstrap_target
            && !(bootstrap_target - self.initial_leaf_count()).is_multiple_of(event_leaf_delta)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive bootstrap-target/initial leaf difference must be divisible by canonical event delta {event_leaf_delta}",
            )));
        }
        if self.topology_interval == 0 || self.max_events_per_interval == 0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive topology interval and event budget must be non-zero".to_string(),
            ));
        }
        if !self.paired_topology_split_radius_scale.is_finite()
            || self.paired_topology_split_radius_scale < 0.0
            || !self.paired_topology_merge_detail_scale.is_finite()
            || self.paired_topology_merge_detail_scale < 0.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive paired-topology scales must be finite and non-negative".to_string(),
            ));
        }
        if self.topology_end_step > 0 && self.topology_end_step < self.topology_start_step {
            return Err(AutomataError::InvalidArgument(
                "adaptive topology end step must not precede its start step".to_string(),
            ));
        }
        if !self.bootstrap_seed_spread.is_finite() || self.bootstrap_seed_spread < 0.0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive bootstrap seed spread must be finite and non-negative".to_string(),
            ));
        }
        if !self.min_topology_footprint_ratio.is_finite()
            || !self.max_topology_footprint_ratio.is_finite()
            || self.min_topology_footprint_ratio <= 0.0
            || self.min_topology_footprint_ratio > 1.0
            || self.max_topology_footprint_ratio < 1.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive topology footprint ratios must satisfy 0 < min <= 1 <= max".to_string(),
            ));
        }
        if !self.render_footprint_relaxation.is_finite()
            || !(0.0..=1.0).contains(&self.render_footprint_relaxation)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive render footprint relaxation must be in [0,1]".to_string(),
            ));
        }
        if !self.split_ratio.is_finite()
            || !self.merge_ratio.is_finite()
            || self.split_ratio <= 0.0
            || self.split_ratio >= 1.0
            || self.merge_ratio <= 1.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive split_ratio must be in (0,1) and merge_ratio > 1".to_string(),
            ));
        }
        if !self.max_unequal_split_measure_ratio.is_finite()
            || self.max_unequal_split_measure_ratio < 1.0
            || self.split_field_neighbors == 0
            || !self.max_neighbor_footprint_ratio.is_finite()
            || (self.max_neighbor_footprint_ratio > 0.0 && self.max_neighbor_footprint_ratio < 1.0)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive continuous split ratio must be >= 1, split-field neighbors must be non-zero, and neighbor grading must be zero or >= 1"
                    .to_string(),
            ));
        }
        if !self.min_reallocation_relative_gain.is_finite()
            || !(0.0..=1.0).contains(&self.min_reallocation_relative_gain)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive min_reallocation_relative_gain must be finite and in [0, 1]".to_string(),
            ));
        }
        if !self.material_seed_bandwidth_exponent.is_finite()
            || self.material_seed_bandwidth_exponent < 0.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive material seed bandwidth exponent must be finite and non-negative"
                    .to_owned(),
            ));
        }
        if !self.material_seed_measure_ratio.is_finite() || self.material_seed_measure_ratio < 1.0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive material seed measure ratio must be finite and at least one".to_owned(),
            ));
        }
        for (name, value) in [
            ("split_probability", self.split_probability),
            ("merge_probability", self.merge_probability),
            ("bandwidth_relaxation", self.bandwidth_relaxation),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive {name} must be finite and in [0,1], got {value}"
                )));
            }
        }
        if !self.merge_extent_ratio.is_finite()
            || self.merge_extent_ratio <= 0.0
            || !self.merge_state_rms_limit.is_finite()
            || self.merge_state_rms_limit <= 0.0
            || !self.split_state_transfer_rms_limit.is_finite()
            || self.split_state_transfer_rms_limit < 0.0
            || !self.split_state_prolongation_scale.is_finite()
            || !(0.0..=1.0).contains(&self.split_state_prolongation_scale)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive merge/state transfer limits are invalid".to_string(),
            ));
        }
        for axis in 0..self.spatial_dims {
            if !self.domain_min[axis].is_finite()
                || !self.domain_max[axis].is_finite()
                || self.domain_min[axis] >= self.domain_max[axis]
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive domain axis {axis} must have finite min < max"
                )));
            }
        }
        Ok(())
    }

    pub const fn supports_bandwidth_adaptation(&self) -> bool {
        matches!(
            self.rule_perception,
            AdaptiveRulePerception::NormalizedAdaptive
        )
    }

    pub fn base_rule_footprint(&self) -> f32 {
        if self.base_rule_footprint > 0.0 {
            self.base_rule_footprint
        } else {
            self.reference_footprint
        }
    }

    pub fn min_render_footprint(&self) -> f32 {
        if self.min_render_footprint > 0.0 {
            self.min_render_footprint
        } else {
            self.min_footprint
        }
    }

    pub fn max_render_footprint(&self) -> f32 {
        if self.max_render_footprint > 0.0 {
            self.max_render_footprint
        } else {
            self.max_footprint
        }
    }

    pub fn render_footprint(&self, material_footprint: f32) -> f32 {
        let ratio = material_footprint.max(f32::MIN_POSITIVE) / self.reference_footprint;
        (self.reference_footprint * ratio.powf(self.render_footprint_exponent))
            .clamp(self.min_render_footprint(), self.max_render_footprint())
    }

    pub fn initial_leaf_count(&self) -> usize {
        if self.initial_leaves == 0 {
            self.target_leaves
        } else {
            self.initial_leaves
        }
    }

    pub fn bootstrap_fine_leaf_count(&self) -> usize {
        if self.bootstrap_fine_leaves == 0 {
            self.bootstrap_target_leaf_count()
        } else {
            self.bootstrap_fine_leaves
        }
    }

    pub fn bootstrap_target_leaf_count(&self) -> usize {
        if self.bootstrap_target_leaves == 0 {
            self.target_leaves
        } else {
            self.bootstrap_target_leaves
        }
    }

    pub fn bootstrap_quadrature_point_count(&self) -> usize {
        if self.bootstrap_quadrature_points == 0 {
            self.coarse_quadrature_points
        } else {
            self.bootstrap_quadrature_points
        }
    }

    pub fn topology_event_budget(&self, step: usize, leaf_count: usize) -> usize {
        if self.bootstrap_end_step > 0
            && step <= self.bootstrap_end_step
            && leaf_count < self.bootstrap_target_leaf_count()
        {
            if self.bootstrap_events_per_interval == 0 {
                self.max_events_per_interval
            } else {
                self.bootstrap_events_per_interval
            }
        } else {
            self.max_events_per_interval
        }
    }

    /// Returns the next reachable material budget for a scheduled restriction.
    ///
    /// Canonical `2d`-child merges remove `2d - 1` visible leaves. The final
    /// interval is clamped to the configured target, whose reachability is
    /// validated against the fine bootstrap population.
    pub fn scheduled_restriction_target(&self, step: usize, leaf_count: usize) -> Option<usize> {
        if self.hierarchical_restriction_step == 0
            || step < self.hierarchical_restriction_step
            || leaf_count <= self.target_leaves
        {
            return None;
        }
        let leaf_delta = self.hierarchical_restriction_leaf_delta_per_interval;
        if leaf_delta == 0 {
            return (step == self.hierarchical_restriction_step).then_some(self.target_leaves);
        }
        let elapsed = step - self.hierarchical_restriction_step;
        if !elapsed.is_multiple_of(self.topology_interval) {
            return None;
        }
        Some(
            leaf_count
                .saturating_sub(leaf_delta)
                .max(self.target_leaves),
        )
    }

    pub fn is_scheduled_restriction_step(&self, step: usize) -> bool {
        if self.hierarchical_restriction_step == 0 || step < self.hierarchical_restriction_step {
            return false;
        }
        let leaf_delta = self.hierarchical_restriction_leaf_delta_per_interval;
        if leaf_delta == 0 {
            return step == self.hierarchical_restriction_step;
        }
        let total_delta = self.bootstrap_fine_leaf_count() - self.target_leaves;
        let intervals = total_delta.div_ceil(leaf_delta);
        let elapsed = step - self.hierarchical_restriction_step;
        elapsed.is_multiple_of(self.topology_interval)
            && elapsed / self.topology_interval < intervals
    }

    pub fn steady_topology_interval(&self) -> usize {
        if self.steady_topology_interval == 0 {
            self.topology_interval
        } else {
            self.steady_topology_interval
        }
    }

    pub fn topology_interval_at(&self, step: usize, leaf_count: usize) -> usize {
        if self.coarse_to_fine_bootstrap_active(step, leaf_count) {
            self.topology_interval
        } else {
            self.steady_topology_interval()
        }
    }

    pub fn steady_topology_start_step(&self) -> usize {
        if self.steady_topology_start_step == 0 {
            self.topology_start_step
        } else {
            self.steady_topology_start_step
        }
    }

    pub fn is_topology_step(&self, step: usize, leaf_count: usize) -> bool {
        if self.topology_end_step > 0 && step > self.topology_end_step {
            return false;
        }
        let start_step = if self.coarse_to_fine_bootstrap_active(step, leaf_count) {
            self.topology_start_step
        } else {
            self.steady_topology_start_step()
        };
        step >= start_step && step.is_multiple_of(self.topology_interval_at(step, leaf_count))
    }

    pub fn coarse_to_fine_bootstrap_active(&self, step: usize, leaf_count: usize) -> bool {
        self.bootstrap_end_step > 0
            && step <= self.bootstrap_end_step
            && leaf_count < self.bootstrap_target_leaf_count()
    }

    pub fn residual_gate(&self, material_footprint: f32) -> f32 {
        let reference = match self.residual_gate_reference {
            AdaptiveResidualGateReference::BaseRule => self.base_rule_footprint(),
            AdaptiveResidualGateReference::TargetBudget => self.reference_footprint,
        };
        ((material_footprint / reference).ln() / std::f32::consts::LN_2).clamp(-3.0, 3.0)
    }

    pub fn is_coarse_rule_footprint(&self, material_footprint: f32) -> bool {
        material_footprint > self.base_rule_footprint() * (1.0 + 32.0 * f32::EPSILON)
    }

    pub fn local_residual_output_scale(&self, output: usize) -> f32 {
        if output < self.spatial_dims {
            self.local_residual_motion_scale
        } else {
            self.local_residual_state_scale
        }
    }

    pub fn split_state_transfer_rms_limit(&self) -> f32 {
        if self.split_state_transfer_rms_limit > 0.0 {
            self.split_state_transfer_rms_limit
        } else {
            self.merge_state_rms_limit
        }
    }
}

const fn default_merge_extent_ratio() -> f32 {
    4.0
}

const fn default_paired_topology_split_radius_scale() -> f32 {
    1.0
}

const fn default_paired_topology_merge_detail_scale() -> f32 {
    0.01
}

const fn legacy_runtime_topology_control() -> AdaptiveTopologyControl {
    AdaptiveTopologyControl::Learned
}

const fn default_local_residual_scale() -> f32 {
    1.0
}

const fn default_min_topology_footprint_ratio() -> f32 {
    0.5
}

const fn default_max_topology_footprint_ratio() -> f32 {
    2.0
}

const fn default_merge_state_rms_limit() -> f32 {
    2.0
}

const fn default_rule_graph_policy() -> AdaptiveGraphPolicy {
    AdaptiveGraphPolicy::RawSupport
}

const fn default_render_transition_steps() -> u32 {
    12
}

const fn default_render_footprint_relaxation() -> f32 {
    0.35
}

const fn default_render_footprint_exponent() -> f32 {
    1.0
}

const fn default_material_seed_measure_ratio() -> f32 {
    1.0
}

const fn default_max_unequal_split_measure_ratio() -> f32 {
    1.0
}

const fn default_split_field_neighbors() -> usize {
    16
}

impl Default for AdaptiveNpaConfig {
    fn default() -> Self {
        Self::growing_2d()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRolloutConfig {
    pub steps: usize,
    pub dt: f32,
    pub update_prob: f32,
    pub seed: u64,
    #[serde(default = "default_true")]
    pub bandwidth_adaptation_enabled: bool,
    #[serde(default = "default_true")]
    pub topology_enabled: bool,
    pub snapshot_interval: usize,
}

const fn default_true() -> bool {
    true
}

impl AdaptiveRolloutConfig {
    pub fn validate(&self) -> AutomataResult<()> {
        if self.steps == 0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive rollout steps must be non-zero".to_string(),
            ));
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive rollout dt must be finite and positive".to_string(),
            ));
        }
        if !self.update_prob.is_finite() || !(0.0..=1.0).contains(&self.update_prob) {
            return Err(AutomataError::InvalidArgument(
                "adaptive rollout update_prob must be finite and in [0,1]".to_string(),
            ));
        }
        if self.snapshot_interval == 0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive snapshot_interval must be non-zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for AdaptiveRolloutConfig {
    fn default() -> Self {
        Self {
            steps: 128,
            dt: 1.0,
            update_prob: 0.5,
            seed: 42,
            bandwidth_adaptation_enabled: true,
            topology_enabled: true,
            snapshot_interval: 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_render_mapping_tracks_material_footprint() {
        let config = AdaptiveNpaConfig::growing_2d();
        for footprint in [
            config.min_footprint,
            config.reference_footprint,
            config.max_footprint,
        ] {
            assert_eq!(config.render_footprint(footprint), footprint);
        }
    }

    #[test]
    fn render_mapping_can_bound_and_emphasize_scale_without_changing_material_limits() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = 0.01;
        config.base_rule_footprint = 0.01;
        config.min_footprint = 0.001;
        config.max_footprint = 0.1;
        config.min_render_footprint = 0.004;
        config.max_render_footprint = 0.04;
        config.render_footprint_exponent = 1.5;
        config.validate().unwrap();

        assert_eq!(config.render_footprint(0.001), 0.004);
        assert_eq!(config.render_footprint(0.01), 0.01);
        assert!(config.render_footprint(0.02) > 0.02);
        assert_eq!(config.render_footprint(0.1), 0.04);
        assert_eq!(config.min_footprint, 0.001);
        assert_eq!(config.max_footprint, 0.1);
    }

    #[test]
    fn bootstrap_and_steady_topology_have_independent_start_steps() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.topology_interval = 1;
        config.steady_topology_interval = 16;
        config.topology_start_step = 1;
        config.steady_topology_start_step = 64;
        config.bootstrap_end_step = 1;
        config.target_leaves = 64;

        assert!(config.is_topology_step(1, 16));
        assert!(!config.is_topology_step(16, 64));
        assert!(!config.is_topology_step(48, 64));
        assert!(config.is_topology_step(64, 64));
    }

    #[test]
    fn topology_end_step_bounds_recurrent_adaptation() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.topology_interval = 8;
        config.topology_start_step = 16;
        config.topology_end_step = 16;

        assert!(!config.is_topology_step(8, config.target_leaves));
        assert!(config.is_topology_step(16, config.target_leaves));
        assert!(!config.is_topology_step(24, config.target_leaves));
    }

    #[test]
    fn reallocation_gain_margin_has_a_bounded_disable_value() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.min_reallocation_relative_gain = 1.0;
        config.validate().unwrap();

        config.min_reallocation_relative_gain = 1.000_001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn fine_to_coarse_curriculum_may_start_above_target_budget() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 1_024;
        config.target_leaves = 3_070;
        config.max_leaves = 4_096;
        config.initial_leaves = 4_096;
        config.bootstrap_fine_leaves = 4_096;
        config.bootstrap_end_step = 0;

        config.validate().unwrap();
        assert_eq!(config.initial_leaf_count(), 4_096);
    }

    #[test]
    fn lod_bootstrap_can_fill_fine_then_restrict_to_steady_budget() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 1_024;
        config.target_leaves = 3_070;
        config.bootstrap_target_leaves = 4_096;
        config.max_leaves = 4_096;
        config.initial_leaves = 1_024;
        config.bootstrap_fine_leaves = 4_096;
        config.topology_start_step = 1;
        config.topology_end_step = 8;
        config.bootstrap_end_step = 8;
        config.bootstrap_events_per_interval = 128;
        config.topology_interval = 1;
        config.hierarchical_restriction_step = 128;
        config.hierarchical_restriction_leaf_delta_per_interval = 96;
        config.coarse_quadrature_points = 2;
        config.bootstrap_quadrature_points = 4;

        config.validate().unwrap();
        assert_eq!(config.bootstrap_target_leaf_count(), 4_096);
        assert_eq!(config.bootstrap_quadrature_point_count(), 4);
        assert!(config.coarse_to_fine_bootstrap_active(1, 1_024));
        assert!(!config.coarse_to_fine_bootstrap_active(9, 4_096));
        assert_eq!(config.topology_event_budget(1, 1_024), 128);
        assert_eq!(config.scheduled_restriction_target(127, 4_096), None);
        assert_eq!(config.scheduled_restriction_target(128, 4_096), Some(4_000));
        assert_eq!(config.scheduled_restriction_target(129, 4_000), Some(3_904));
        assert_eq!(config.scheduled_restriction_target(138, 3_136), Some(3_070));
        assert_eq!(config.scheduled_restriction_target(139, 3_070), None);
        assert!(config.is_scheduled_restriction_step(128));
        assert!(config.is_scheduled_restriction_step(138));
        assert!(!config.is_scheduled_restriction_step(139));
    }

    #[test]
    fn zero_restriction_event_budget_preserves_one_shot_schedule() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 40;
        config.target_leaves = 40;
        config.initial_leaves = 64;
        config.max_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.hierarchical_restriction_step = 7;

        config.validate().unwrap();
        assert_eq!(config.scheduled_restriction_target(6, 64), None);
        assert_eq!(config.scheduled_restriction_target(7, 64), Some(40));
        assert_eq!(config.scheduled_restriction_target(8, 64), None);
    }

    #[test]
    fn residual_output_groups_have_independent_backward_compatible_gains() {
        let mut config = AdaptiveNpaConfig::growing_2d();
        assert_eq!(config.local_residual_output_scale(0), 1.0);
        assert_eq!(config.local_residual_output_scale(2), 1.0);

        config.local_residual_motion_scale = 0.25;
        config.local_residual_state_scale = 1.5;
        config.validate().unwrap();

        assert_eq!(config.local_residual_output_scale(0), 0.25);
        assert_eq!(config.local_residual_output_scale(1), 0.25);
        assert_eq!(config.local_residual_output_scale(2), 1.5);
    }
}
