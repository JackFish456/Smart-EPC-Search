from __future__ import annotations

from dataclasses import dataclass

from epc_smart_search.answer_policy import AnswerPolicy
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.retrieval import HashingEmbedder, HybridRetriever
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, pack_vector


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    question: str
    chunks: tuple[ChunkRecord, ...]
    expected_snippets: tuple[str, ...]


class FakeGemma:
    def ask(self, *args, **kwargs):
        raise AssertionError("Gemma should not be called for benchmarked grounded answers.")


def test_bad_question_benchmark_cases() -> None:
    cases = (
        BenchmarkCase(
            name="dew_point_configuration",
            question="What is the dew point configuration?",
            chunks=(
                _chunk(
                    "generic_dew",
                    "3.1",
                    "Dew Point",
                    "Dew point testing shall be completed during commissioning.",
                    10,
                ),
                _chunk(
                    "config_clause",
                    "7.4.2",
                    "Fuel Gas Dew Point Configuration",
                    "The fuel gas dew point configuration shall use a duplex analyzer arrangement with automatic switchover.",
                    41,
                ),
            ),
            expected_snippets=("duplex analyzer arrangement with automatic switchover", "Section 7.4.2"),
        ),
        BenchmarkCase(
            name="dew_point_heater_configuration",
            question="dew point heater configuration",
            chunks=(
                _chunk(
                    "heater_config",
                    "5.23",
                    "Dew Point Heater",
                    "The dew point heater controls are configured as 4x50% electric heaters to achieve the minimum desired gas fuel temperature at the heater outlet based on a required temperature differential.",
                    2686,
                ),
            ),
            expected_snippets=('"4x50%"', "Section 5.23"),
        ),
        BenchmarkCase(
            name="selected_turbine",
            question="What is the turbine we are using?",
            chunks=(
                _chunk(
                    "generic_turbine",
                    "9.1",
                    "Steam Turbine",
                    "Turbine components shall comply with the applicable standards for rotating equipment.",
                    18,
                ),
                _chunk(
                    "selected_turbine",
                    "9.2",
                    "Selected Turbine Generator",
                    "The selected turbine model shall be Siemens SGT6-5000F for the project.",
                    19,
                ),
            ),
            expected_snippets=("Siemens SGT6-5000F", "Section 9.2"),
        ),
        BenchmarkCase(
            name="design_conditions",
            question="What are the compressor design conditions?",
            chunks=(
                _chunk(
                    "generic_design",
                    "12.1",
                    "General Design Philosophy",
                    "Design reviews and operating conditions shall be coordinated with the project team.",
                    27,
                ),
                _chunk(
                    "compressor_conditions",
                    "12.4.1",
                    "Compressor Design Conditions",
                    "The compressor design conditions shall be 1250 psig discharge pressure and 105 degF inlet temperature.",
                    28,
                ),
            ),
            expected_snippets=("1250 psig discharge pressure", "105 degF inlet temperature"),
        ),
        BenchmarkCase(
            name="closed_cooling_water_pump_configuration_with_heading_only_noise",
            question="what is the closed cooling water pump configuration",
            chunks=(
                _chunk(
                    "ccw_heading_only",
                    "3",
                    "Closed Cooling Water Pump",
                    "3",
                    1485,
                ),
                _chunk(
                    "ccw_system",
                    "5.18",
                    "Closed Cooling Water",
                    "5.18 Closed Cooling Water Function The function of the Closed Cooling Water System is to transfer heat from the various plant equipment and fin fans to the atmosphere via the air cooler. Major Components Two 100% capacity, horizontal, closed cooling water pumps with piping valves and expansion joints. Two full capacity closed cooling water horizontal pumps with motors. One closed cooling water system head tank.",
                    261,
                ),
            ),
            expected_snippets=("Two 100% capacity", "Section 5.18"),
        ),
        BenchmarkCase(
            name="generic_design_temperatures_prefers_general_guidance_over_random_values",
            question="what are the design temperatures",
            chunks=(
                _chunk(
                    "design_temp_general",
                    "4.2.2.2",
                    "Design Temperature and Pressure",
                    "The design pressure and temperature for piping will be based on the operating pressures and temperatures established above with design margins as defined below. Note different segments of a system can have different design pressures and temperatures, therefore engineering analysis of different segments is required so the segments are not over designed, under designed, or over pressurized. The systems will be designed and specified based on the design pressures and temperatures determined.",
                    181,
                ),
                _chunk(
                    "design_temp_table",
                    "4.2.2.2A",
                    "Design Data",
                    "HP Steam 1450 psig 1050 degF. LP Steam 220 psig 450 degF.",
                    182,
                ),
            ),
            expected_snippets=("design pressure and temperature for piping will be based on the operating pressures and temperatures", "Section 4.2.2.2"),
        ),
        BenchmarkCase(
            name="fire_water_pump_horsepower",
            question="what is the fire water pump horse power",
            chunks=(
                _chunk(
                    "random_schedule",
                    "6.9",
                    "KV - 480 V",
                    "02-CCW-PMP-01A TRAIN 2 CLOSED COOLING WATER PUMP MOTOR 1A 1200 01-BFW-PMP-01A TRAIN 1 BOILER FEEDWATER PUMP MOTOR 1A 4750 HP.",
                    359,
                ),
                _chunk(
                    "fire_water_pump",
                    "8.4.2",
                    "Fire Water Pump",
                    "Each fire water pump shall be rated at 350 HP for the project fire water service.",
                    412,
                ),
            ),
            expected_snippets=("350 HP", "Section 8.4.2"),
        ),
        BenchmarkCase(
            name="appendix_e_emission_guarantees",
            question="what are my emission guarantees",
            chunks=(
                _chunk(
                    "generic_emissions",
                    "7.5.3",
                    "Air Permit Test Result",
                    "The submission must include all emissions values and supporting calculations.",
                    83,
                ),
                _chunk(
                    "appendix_e",
                    "E",
                    "APPENDIX E - Emission Guarantees",
                    "Emission Guarantees.",
                    2686,
                    chunk_type="exhibit",
                ),
                _chunk(
                    "appendix_e_limits",
                    "E.1",
                    "Guarantee Limits",
                    "Seller guarantees NOx emissions shall not exceed 2.0 ppmvd at 15% oxygen. CO emissions shall not exceed 4.0 ppmvd at 15% oxygen.",
                    2687,
                    parent_chunk_id="appendix_e",
                ),
            ),
            expected_snippets=("APPENDIX E - Emission Guarantees", "NOx emissions shall not exceed 2.0 ppmvd", "CO emissions shall not exceed 4.0 ppmvd"),
        ),
        BenchmarkCase(
            name="appendix_e_emission_guarantees_typo",
            question="what are my emission guarentees in appendix e",
            chunks=(
                _chunk(
                    "generic_emissions",
                    "7.5.3",
                    "Air Permit Test Result",
                    "The submission must include all emissions values and supporting calculations.",
                    83,
                ),
                _chunk(
                    "appendix_e",
                    "E",
                    "APPENDIX E - Emission Guarantees",
                    "Emission Guarantees.",
                    2686,
                    chunk_type="exhibit",
                ),
                _chunk(
                    "appendix_e_limits",
                    "E.1",
                    "Guarantee Limits",
                    "Seller guarantees NOx emissions shall not exceed 2.0 ppmvd at 15% oxygen. CO emissions shall not exceed 4.0 ppmvd at 15% oxygen.",
                    2687,
                    parent_chunk_id="appendix_e",
                ),
            ),
            expected_snippets=("APPENDIX E - Emission Guarantees", "NOx emissions shall not exceed 2.0 ppmvd", "CO emissions shall not exceed 4.0 ppmvd"),
        ),
    )

    for case in cases:
        answer = _run_case(case)
        assert not answer.refused, case.name
        for snippet in case.expected_snippets:
            assert snippet in answer.text, f"{case.name}: missing {snippet!r} in {answer.text!r}"


def _run_case(case: BenchmarkCase):
    retriever = _seed_retriever(list(case.chunks))

    class BenchmarkAssistant:
        pass

    assistant = BenchmarkAssistant()
    assistant.retriever = retriever
    assistant.gemma = FakeGemma()
    assistant.answer_policy = AnswerPolicy(retriever.store, retriever)
    return assistant.answer_policy.answer(case.question, None, assistant.gemma)


def _seed_retriever(chunks: list[ChunkRecord]) -> HybridRetriever:
    db_path = "file:bad_question_benchmark?mode=memory&cache=shared"
    store = ContractStore(db_path)
    embedder = HashingEmbedder(dimension=24)
    pages = [PageText(page_num=chunk.page_start, text=chunk.full_text, ocr_used=False) for chunk in chunks]
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=len(pages),
        chunks=chunks,
        pages=pages,
        features=build_chunk_features(chunks),
        embeddings={chunk.chunk_id: pack_vector(embedder.embed(chunk.full_text)) for chunk in chunks},
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    return HybridRetriever(store, embedder)


def _chunk(
    chunk_id: str,
    section_number: str,
    heading: str,
    full_text: str,
    page_num: int,
    *,
    chunk_type: str = "section",
    parent_chunk_id: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc1",
        chunk_type=chunk_type,
        section_number=section_number,
        heading=heading,
        full_text=full_text,
        page_start=page_num,
        page_end=page_num,
        parent_chunk_id=parent_chunk_id,
        ordinal_in_document=page_num,
    )
