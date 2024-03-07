import ecto_ds.sql as sql
import cam_smolt_quality.configs as cfg
import pandas as pd

LICENSEE_ID = 117
START_DATE = "2017-01-01" if cfg.PIPELINE_TYPE == 'train' else pd.to_datetime(
    cfg.CURRENT_DATE, utc=True) - pd.DateOffset(years=2)
MAX_DATE = pd.to_datetime(cfg.CURRENT_DATE, utc=True)

config = {
    "extract": {
        "ROOT_DIR": cfg.ROOT_DIR,
        "RAW_DATA_SUB_DIR": "data/raw",
        "CURRENT_DATE": cfg.CURRENT_DATE,
        "READ_PARAMS": cfg.READ_PARAMS,
        "PIPELINE_TYPE": cfg.PIPELINE_TYPE,
        "csv_buffer": cfg.CSV_BUFFER,
        "save_raw_files": True if cfg.RUN_ENV == "dev" else False,
    },
    "loader": {
        "stockings_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.stockings.query,
                # 'to_locus_id', 'to_locus_name', 'to_lp_id', 'from_lp_id',
                "keys": ['from_lp_id', 'to_lp_id', "transfer_date", "to_site_name", "to_fish_group", ],
                "values": ["stock_volume", "stock_cnt", "stock_bms"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "vaccines_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.vaccination.query,
                "keys": ["event_date", "site_name", "fish_group", "manufacturer", "type", "method", "expiration_date"],
                "values": ["dose", "fish_cnt"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "sw_mortality_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.mortality.query,
                "keys": ["fish_group", "site_name", "event_date", "mortality_reason",],
                "values": ["mortality_cnt"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    # "MORTALITY_REASONS_TO_EXCLUDE": ('')
                },
            }
        },
        "fw_mortality_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.mortality.query,
                # "mortality_reason",
                "keys": ["fish_group", "site_name", "fw_locus_prefix", "event_date", ],
                "values": ["mortality_cnt"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                    # "MORTALITY_REASONS_TO_EXCLUDE": ('')
                },
            }
        },
        "movements_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.movements.query,
                # "src_lp_id", "dst_lp_id",  "dst_locus_name", "dst_locus_id", "src_locus_name", "src_locus_id",
                "keys": ["src_fish_group", "dst_fish_group", "src_site_name", "dst_site_name", "movement_type", "event_date"],
                "values": ["transferred_cnt"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "movements_ratio_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.movements_ratio.query,
                # "src_lp_id", "dst_lp_id",  "dst_locus_name", "dst_locus_id", "src_locus_name", "src_locus_id",
                "keys": ['historic_lp_id', 'final_lp_id',],
                "values": ['cnt_ratio', 'bms_ratio'],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                },
            }
        },


        "fw_inventory_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.inventory.query,
                "keys": ['locus_id', 'locus_name', 'fw_locus_prefix', "site_name", "fish_group", "site_type", "event_date"],
                "values": ["start_fish_cnt", "end_fish_cnt", "start_fish_bms", "end_fish_bms", "fish_wg", "degree_days"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction', 'Seawater'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                },
            }
        },
        "sensors_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.sensors.query,
                "keys": ['site_name', 'event_date', 'sensor_type_name', 'sensor_name'],
                "values": ["value", ],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction', 'Seawater'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R',  'Rio Petrohue', 'UPS',)
                },
            }
        },
        
        
        "sw_inventory_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.inventory.query,
                "keys": ['locus_name', 'locus_id', "site_name", "fish_group", "site_type", "event_date"],
                "values": ["start_fish_cnt", "end_fish_cnt", "start_fish_bms", "end_fish_bms", "fish_wg", "degree_days"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Seawater', ''),
                },
            }
        },
        "locus_to_fish_group": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.locus_to_fish_group.query,
                "keys": ["locus_id", "fw_locus_prefix", "fish_group"],
                "values": [],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                },
            }
        },
        "lab_atpasa_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.lab_atpasa.query,
                "keys": ["site_name", "locus_id", "fish_group", "event_date"],
                "values": ["weight", "length", "k_factor", "atpasa", "n_samples"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction', 'Seawater'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                },
            }
        },
        "lab_luf_fish": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.lab_luf_fish.query,
                "keys": ["site_name", "locus_id", "event_date"],
                "values": ["value_ppb", "n_samples"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction', 'Seawater'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                },
            }
        },
        "treatments_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.treatments.query,
                "keys": ["site_name", "fish_group", "event_date", "treatment_method_name", "active_substance_name"],
                "values": ["amount"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS')
                },
            }
        },
        "sw_feed_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.feed.query,
                "keys": ["site_name", "locus_name", "locus_id", "fish_group", "event_date"],
                "values": ["feed_amount"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "fw_feed_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.feed.query,
                "keys": ["site_name", "fish_group", "event_date", "feed_name"],
                "values": ["feed_amount"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                },
            }
        },
        "light_regime_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.light_regime.query,
                "keys": ["event_date", "site_name",  "locus_id", "fw_locus_prefix", "light_regime"], 
                "values": [],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',),
                    #"LIGHT_REGIMES": ('Invierno', 'Verano')
                },
            }
        },
        "locus_to_locus_group_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.locus_to_locus_group.query,
                "keys": ["site_name", "locus_id", "locus_name", "fw_locus_prefix", "locus_group_id", "locus_group_name"],
                "values": [],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',)
                },
            }
        },
        "logbook_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.logbook.query,
                "keys": ["event_date",'site_name', "locus_group_id", "locus_group_name", "variable"],
                "values": ["value"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,  # '2021-12-30', #Date of the first job
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS'),
                },
            }
        },
        "jobs_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.jobs.query,
                "keys": ["event_date", "site_name", "locus_group_id", "locus_group_name", "variable"],
                "values": ["value",],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS'),
                },
            }
        },
        "cam_fishgroups": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).views.cam_fishgroups.query,
                "keys": ["fish_group", "specie_name", "season", "strain_name", "generation", "order_num"],
                "values": [],
                "args": {
                },
            }
        },
        "site_map": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mapping.site.query,
                "keys": ["site_id", "site_name"],
                "values": [],
                "args": {
                    "LICENSEE_ID": 117
                },
            }
        },
        "fish_groups_map": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mapping.fish_group.query,
                "keys": ["fish_group_id", "fish_group"],
                "values": [],
                "args": {
                    "LICENSEE_ID": 117
                },
            }
        }, 
     
    },
    "slack": {
        "channel": "#dst-notifications"
    },
    "modeling": {
        "forecast": {
            "model_id": 90,
            "target_name": "mrt:post_7d:perc",
            "model_name": 'cb_posttreatmrt_w0_v1.0'
        }
    },
}

SPANISH_MONTHS_MAP = {
    'C1': 'Summer',
    'C2': 'Autumn',
    'C3': 'Winter',
    'C4': 'Spring',
    'Verano': 'Summer',
    'Otoño': 'Autumn',
    'Invierno': 'Winter',
    'Primavera': 'Spring',
    '1': 'Summer',
    '2': 'Autumn',
    '3': 'Winter',
    '4': 'Spring',
}

SRC_PROJECT_MANUAL_MAP = {
    'SNFAN1500': 'SNFAN1600',
    'SNFAN1600': 'SNFAN1700',
    'SNFLY1600': 'SNFLY1700',
    'SNFLY1700': 'SNFLY1800',
    'SNFLY1800': 'SNFLY1900',
    'SNAQG1900': 'SNAQG1800',
    'SNFLY1900': 'SNFLY2000',
    'SNLYF2100': 'SNFLY2100',
    'SNAQG2100': 'SNAQG2200',
    'SNAQG2200': 'SNAQG2300'
}


STRAIN_REPLACE_MAP = {'LYF': 'FLY'}


MRTPERC_GROUPS = {
    'very low': 0.0,
    'low': 0.430,
    'medium': 0.948,
    'high': 1.508,
    'extreme': 3.055
} #Histroical q = [0, 0.05, 0.35, 0.65, 0.95, 1]

