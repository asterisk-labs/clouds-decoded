from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, ClassVar, Dict, List, Optional
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseProcessorConfig(BaseModel):
    """
    Base configuration class for all processors.
    Utilizes Pydantic for validation and type checking.
    """
    model_config = ConfigDict(extra='forbid')

    # Fields that control *performance* or *placement* but not the
    # semantic result of a step. Excluded from config hashing and stored
    # provenance so that a scene processed on CPU today and re-run on
    # GPU tomorrow is not treated as "stale config". Subclasses may
    # extend this set (e.g. batch_size, n_workers) but should never
    # *remove* entries without careful consideration.
    NON_SEMANTIC_FIELDS: ClassVar[frozenset] = frozenset({"device"})

    @classmethod
    def _semantic_dump(cls, instance: "BaseProcessorConfig") -> Dict[str, Any]:
        """Return a ``model_dump(mode='json')`` stripped of non-semantic keys.

        Used by the project-level config hash and the file-provenance
        integrity check. Computed fields are excluded as well, since
        they are derived from other fields and would duplicate entries
        in the hash input.
        """
        computed = type(instance).model_computed_fields
        exclude = set(cls.NON_SEMANTIC_FIELDS)
        if computed:
            exclude |= set(computed.keys())
        return instance.model_dump(mode='json', exclude=exclude)
    output_dir: Optional[str] = Field(None, description="Directory to save outputs")
    working_resolution: Optional[int] = Field(
        default=None,
        ge=10,
        description=(
            "Resolution in metres at which inference is performed. "
            "None = processor's natural resolution."
        ),
    )
    output_resolution: Optional[int] = Field(
        default=None,
        ge=10,
        description=(
            "Output resolution in metres. When set, the result from process() is "
            "resampled to this resolution before being returned. None = return at "
            "the processor's working resolution."
        ),
    )

    @model_validator(mode='after')
    def _check_resolution_ordering(self) -> 'BaseProcessorConfig':
        """Reject configs where output_resolution < working_resolution."""
        if (
            self.working_resolution is not None
            and self.output_resolution is not None
            and self.output_resolution < self.working_resolution
        ):
            raise ValueError(
                f"output_resolution ({self.output_resolution}) must be >= "
                f"working_resolution ({self.working_resolution})"
            )
        return self

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> 'BaseProcessorConfig':
        """Load configuration from a YAML file.

        Args:
            config_path: Path to a YAML file. If ``None``, returns a
                default-constructed config instance.

        Returns:
            An instance of the config subclass populated from the file,
            or defaults if *config_path* is ``None``.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
        """
        if config_path is None:
            return cls()

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # If the yaml is empty
        if data is None:
            data = {}
            
        return cls(**data)

    def to_yaml(self, config_path: str) -> None:
        """Write configuration to a YAML file.

        Computed fields (@computed_field) are excluded since they are derived
        from other fields and should not be edited directly.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        computed = type(self).model_computed_fields
        exclude = set(computed.keys()) if computed else set()
        # Use mode='json' to ensure tuples are converted to lists, 
        # avoiding !!python/tuple tags that safe_load cannot handle.
        data = self.model_dump(exclude=exclude, mode='json')
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
