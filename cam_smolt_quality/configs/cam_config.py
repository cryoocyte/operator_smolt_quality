import ecto_ds.sql as sql
import cam_smolt_quality.configs as cfg
import pandas as pd

LICENSEE_ID = 117
START_DATE = "2017-01-01" if cfg.PIPELINE_TYPE == 'train' else pd.to_datetime(cfg.CURRENT_DATE, utc=True) - pd.DateOffset(years=2)
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
                "keys": ["transfer_date", "to_site_name", "to_locus_id", "to_locus_name", "to_lp_id", "from_lp_id", "fish_group"],
                "values": ["stock_cnt"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    #"FW_SITES": () #'UPS', 
                },
            }
        },
        "mortality_transfer_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.mortality_transfer.query,
                "keys": ["fish_group", "locus_id", "locus_name", "site_name", "lp_id", "event_date",],
                "values": ["mortality_count"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Seawater', ''),
                    #"SITES_TO_INCLUDE": ('Rio Petrohue', 'UPS', ), #'65.100224.RD Este R', 'Polcura Rep', 'Transport', 
                    "MORTALITY_REASONS_TO_EXCLUDE": ('Transporte', '')
                },
            }
        },
        
        "movements_ratio_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.movements_ratio.query,
                "keys": ["historic_lp_id", "final_lp_id",],
                "values": ["cnt_ratio", "bms_ratio"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                },
            }
        }, 
        "inventory_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.inventory.query,
                "keys": ["locus_id", "locus_name", "site_name", "lp_id", "fish_group", "year_class", "fg_order", "event_date"],
                "values": ["start_fish_cnt", "end_fish_cnt", "start_fish_bms", "end_fish_bms", "fish_wg"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    #"SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS') 
                },
            }
        },
        "lab_atpasa_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.lab_atpasa.query,
                "keys": ["locus_id", "locus_name", "site_name", "lp_id", "fish_group", "year_class", "fg_order", "event_date"],
                "values": ["weight", "length", "k_factor", "atpasa",],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "treatments_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.treatments.query,
                "keys": ["locus_name", 'locus_id', "site_name", "lp_id", "fish_group", "event_date", "treatment_method_name", "active_substance_name"],
                "values": ["amount",],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    #"SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS') 
                },
            }
        },
        "feed_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.feed.query,
                "keys": ["locus_name", 'locus_id', "site_name", "lp_id", "fish_group", "event_date", "feed_name"],
                "values": ["feed_amount"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    #"SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS') 
                },
            }
        },
        "light_regime_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.light_regime.query,
                "keys": ["locus_name", 'locus_id', "site_name", "lp_id", "fish_group", "event_date", "type_name"],
                "values": [],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "locus_to_locus_group_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.locus_to_locus_group.query,
                "keys": ["locus_id", "locus_group_id"],
                "values": [],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                },
            }
        },
        "logbook_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.logbook.query,
                "keys": ["event_date", "locus_group_id", "variable"],
                "values": ["value"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE, #'2021-12-30', #Date of the first job 
                    #"SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',  'Polcura Rep'),
                },
            }
        },
        "jobs_test_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.jobs_test.query,
                "keys": ["event_date", "locus_group_id", "variable"],
                "values": ["value",],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    #'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                },
            }
        },
        "mortality_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.mortality.query,
                "keys": ["fish_group", "locus_id", "locus_name", "site_name", "lp_id", "event_date", "mortality_reason"],
                "values": ["mortality_count"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
                    "SITE_TYPE": ('Freshwater', 'Freshwater Reproduction'),
                    #"SITES_TO_INCLUDE": ('Rio Petrohue', 'UPS', ), #'65.100224.RD Este R', 'Polcura Rep', 'Transport', 
                    #"MORTALITY_REASONS": ('Opérculo Corto', 'Eliminación Opérculo', 'Eliminación Productiva')
                },
            }
        },
        "logbook_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.logbook.query,
                "keys": ["event_date", "locus_group_id", "variable"],
                "values": ["value"],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE, #'2021-12-30', #Date of the first job 
                    #"SITES_TO_INCLUDE": ('65.100224.RD Este R', 'Transport', 'Rio Petrohue', 'UPS',  'Polcura Rep'),
                },
            }
        },
        "jobs_test_mrts": {
            "sql": {
                "query": getattr(sql, cfg.RUN_ENV).mrts.jobs_test.query,
                "keys": ["event_date", "locus_group_id", "variable"],
                "values": ["value",],
                "args": {
                    "LICENSEE_ID": LICENSEE_ID,
                    #'START_DATE': START_DATE,
                    "MAX_DATE": MAX_DATE,
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
