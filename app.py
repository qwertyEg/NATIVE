import streamlit as st
import os
import tempfile
from logic import Interaction_with_LLM, generate_word_document, convert_to_wav_16k, wav_recognition_with_chunk

st.set_page_config(page_title="На родном. Татарча.", layout="centered")

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'lesson_data' not in st.session_state:
    st.session_state.lesson_data = {}
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'ex_checked' not in st.session_state:
    st.session_state.ex_checked = False

def next_step():
    st.session_state.step += 1
    st.session_state.ex_checked = False

st.markdown("## На родном. Татарча.")
st.markdown("Персонализированные уроки на основе живых аудио и видеоматериалов: загружайте свои файлы или выбирайте готовые ролики по интересам — сервис сам превратит их в цепочку понятных упражнений, чтобы вы постепенно и уверенно осваивали татарский язык.")
st.download_button("Скачать пример урока (PDF)", b"PDF", file_name="example.pdf")
st.divider()

if st.session_state.step == 0:
    st.chat_message("assistant").write("Здравствуйте! Исәнмесез! Я помогу вам превратить аудио и видеоматериалы в персональный урок татарского языка. Нажмите кнопку „Алга!“, чтобы начать.")
    st.button("Алга!", on_click=next_step)

elif st.session_state.step == 1:
    st.chat_message("assistant").write("Выберите, с чего начать:\n— Загрузите свой файл (аудио или видео);\n— Вставьте ссылку на ролик;\n— Или выберите один из готовых материалов по интересам.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Загрузить свой материал"):
            st.session_state.upload_mode = 'custom'
            next_step()
            st.rerun()
    with col2:
        if st.button("Выбрать из готовых материалов"):
            st.session_state.upload_mode = 'ready'
            next_step()
            st.rerun()

elif st.session_state.step == 2:
    if st.session_state.upload_mode == 'custom':
        st.info("Вы можете загрузить аудиофайл в форматах MP3 или WAV. Максимальная длительность — до 3 минут, более длинные записи будут автоматически обрезаны до первых 3 минут. Максимальный размер файла — до 20 МБ.")
        uploaded_file = st.file_uploader("Загрузить файл", type=['mp3', 'wav'])
        
        if uploaded_file is not None:
            with st.spinner("Обрабатываю ваш материал… Это может занять до минуты."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
                    tmp_in.write(uploaded_file.getvalue())
                    tmp_in_path = tmp_in.name
                
                tmp_out_path = tmp_in_path.replace(".wav", "_16k.wav")
                
                try:
                    convert_to_wav_16k(tmp_in_path, tmp_out_path)
                    raw_text = wav_recognition_with_chunk(tmp_out_path)
                    
                    llm = Interaction_with_LLM()
                    st.session_state.lesson_data['refined_text'] = llm.create_refined(raw_text)
                    st.session_state.lesson_data['frank_text'] = llm.create_i_frank_lesson()
                    st.session_state.lesson_data['collocations'] = llm.create_collocations_lesson()
                    st.session_state.lesson_data['matching'] = llm.create_matching_lesson()
                    st.session_state.lesson_data['gap_fill'] = llm.create_gap_fill_lesson()
                    st.session_state.lesson_data['logical'] = llm.create_logical_completion_lesson()
                    st.session_state.lesson_data['grammar'] = llm.create_grammar_digest_lesson()
                    
                    next_step()
                    st.rerun()
                except Exception as e:
                    st.error("Что‑то пошло не так при загрузке файла. Попробуйте ещё раз или используйте другую запись.")
                finally:
                    if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
                    if os.path.exists(tmp_out_path): os.remove(tmp_out_path)

elif st.session_state.step == 3:
    st.info("Сейчас вы знакомитесь с живым татарским текстом с подсказками-переводами. Просто читайте текст и в случае трудностей обращайтесь к переводу — так вы естественно привыкаете к языку, расширяете словарный запас и начинаете понимать фразы целиком, а не по одному слову.")
    st.write(st.session_state.lesson_data.get('frank_text', ''))
    st.button("Перейти к следующему упражнению", on_click=next_step)

elif st.session_state.step == 4:
    st.info("Ниже вы найдёте самые важные слова и выражения из текста. Освоив этот небольшой набор лексики, вы сможете увереннее понимать и использовать татарский в похожих ситуациях.")
    st.write(st.session_state.lesson_data.get('collocations', '').split('|||')[0])
    st.write("Отлично! Чем лучше вы знаете эти слова, тем увереннее будете чувствовать себя в следующих заданиях.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Перейти к следующему упражнению", on_click=next_step)
    with col2:
        st.button("Пропустить упражнение", on_click=next_step)

elif st.session_state.step == 5:
    st.info("Сейчас вы тренируете быстрый отклик: сопоставляя фразы на татарском с переводом, вы закрепляете значения и учитесь узнавать знакомые выражения „на лету“ — так понимание речи становится проще и увереннее.")
    st.write(st.session_state.lesson_data.get('matching', '').split('|||')[0])
    
    if not st.session_state.ex_checked:
        st.text_input("Ваш ответ", key="ans_match")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Проверить", key="chk_match"):
                st.session_state.score += 1
                st.session_state.ex_checked = True
                st.rerun()
        with col2:
            st.button("Пропустить упражнение", on_click=next_step, key="skip_match")
    else:
        st.success("Отличная работа! Ваш ответ учтен.")
        st.button("Перейти к следующему упражнению", on_click=next_step, key="next_match")

elif st.session_state.step == 6:
    st.info("Это задание помогает довести новые слова до автоматизма. Подставляя пропущенные слова в нужный контекст, вы проверяете себя и учитесь правильно использовать лексику и грамматику в реальных фразах.")
    st.write(st.session_state.lesson_data.get('gap_fill', '').split('|||')[0])
    
    if not st.session_state.ex_checked:
        st.text_input("Ваш ответ", key="ans_gap")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Проверить", key="chk_gap"):
                st.session_state.score += 1
                st.session_state.ex_checked = True
                st.rerun()
        with col2:
            st.button("Пропустить упражнение", on_click=next_step, key="skip_gap")
    else:
        st.success("Отличная работа! Ваш ответ учтен.")
        st.button("Перейти к следующему упражнению", on_click=next_step, key="next_gap")

elif st.session_state.step == 7:
    st.info("Здесь вы переходите от пассивного понимания к активной речи. Продолжая предложения, вы пробуете формулировать мысли по-татарски, а сервис мягко подскажет, насколько естественно звучит ваш вариант.")
    st.write(st.session_state.lesson_data.get('logical', '').split('|||')[0])
    
    if not st.session_state.ex_checked:
        st.text_input("Ваш ответ", key="ans_log")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Проверить", key="chk_log"):
                st.session_state.score += 1
                st.session_state.ex_checked = True
                st.rerun()
        with col2:
            st.button("Пропустить упражнение", on_click=next_step, key="skip_log")
    else:
        st.success("Здорово! Вы пробуете выражать свои мысли по-татарски — с каждым таким шагом речь становится увереннее.")
        st.button("Перейти к следующему упражнению", on_click=next_step, key="next_log")

elif st.session_state.step == 8:
    st.info("Сейчас вы разбираете самые важные грамматические конструкции из текста. Короткое объяснение и примеры помогут понять, как устроена фраза по-татарски, чтобы вы могли строить такие же предложения сами — увереннее и без лишних ошибок.")
    st.write("Грамматика из этого текста")
    st.write(st.session_state.lesson_data.get('grammar', '').split('|||')[0])
    st.write("Чем лучше вы понимаете, как устроены эти конструкции, тем легче строить свои предложения без ошибок.")
    
    st.button("Пропустить упражнение", on_click=next_step, key="skip_gram")

elif st.session_state.step == 9:
    st.info("На этом шаге вы переходите к живому общению: аватар разговаривает с вами по-татарски по теме урока, задаёт вопросы и мягко исправляет ошибки. Это безопасное пространство, где можно спокойно тренировать речь, пробовать новые фразы и становиться увереннее в разговоре.")
    
    if 'avatar_chat' not in st.session_state:
        st.session_state.avatar_chat = [{"role": "assistant", "content": "Сәлам! Әйдә, дәрес темасына сөйләшик. (Привет! Давай поговорим по теме урока.)"}]
        st.session_state.chat_llm = Interaction_with_LLM()

    for msg in st.session_state.avatar_chat:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Напишите сообщение по-татарски..."):
        st.session_state.avatar_chat.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Аватар печатает..."):
            reply = st.session_state.chat_llm.chat_with_avatar(prompt)
            st.session_state.avatar_chat.append({"role": "assistant", "content": reply})
            st.rerun()

    st.button("Завершить разговор и перейти к минитесту", on_click=next_step)

elif st.session_state.step == 10:
    st.write(f"Вы набрали {st.session_state.score} баллов за упражнения.")
    st.write("Ваше условное место в рейтинге активности: 12.")
    st.write("Вы прошли все этапы урока — от текста до диалога. Вы – молодец! Можете вернуться к новым материалам или повторить похожие задания по этой теме.")
    
    word_file = generate_word_document(st.session_state.lesson_data)
    st.download_button(
        label="Скачать весь урок в формате Word",
        data=word_file,
        file_name="tatar_lesson.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )