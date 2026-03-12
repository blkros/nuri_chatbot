# 그룹웨어 연동 API 요청

AI 챗봇에서 호출할 API 3개입니다. 응답은 JSON으로 부탁드립니다.

---

## 1. 빈 회의실 조회

`GET /api/meeting-rooms/available?date=2026-03-10&start_time=14:00&end_time=15:00`

```json
{
  "rooms": [
    {
      "room_name": "3층 대회의실",
      "booker_name": null,
      "meeting_title": null,
      "start_time": null,
      "end_time": null
    },
    {
      "room_name": "2층 소회의실A",
      "booker_name": "김철수",
      "meeting_title": "주간 스크럼",
      "start_time": "13:30",
      "end_time": "14:30"
    }
  ]
}
```

파라미터: `date`(필수), `start_time`(필수), `end_time`(필수), `floor`(선택)
빈 회의실은 booker_name/meeting_title/start_time/end_time이 null

---

## 2. 직원 검색

`GET /api/employees/search?query=김철수`

```json
{
  "employees": [
    {
      "name": "김철수",
      "department": "개발팀",
      "position": "선임연구원",
      "role": "백엔드 개발",
      "mobile": "010-1234-5678",
      "phone": "031-123-3421",
      "email": "cskim@company.com"
    }
  ],
  "total": 1
}
```

파라미터: `query`(필수, 이름/부서/직급), `department`(선택), `limit`(선택, 기본 10)

---

## 3. 회사 뉴스 / 공지사항

`GET /api/news?limit=5`

```json
{
  "articles": [
    {
      "title": "2026년 상반기 조직개편 안내",
      "category": "공지",
      "author": "경영지원팀",
      "published_at": "2026-03-08",
      "summary": "3월 10일부로 개발본부 산하 AI팀 신설..."
    }
  ],
  "total": 1
}
```

파라미터: `keyword`(선택), `category`(선택: 공지/소식/인사/행사), `limit`(선택, 기본 5), `start_date`(선택)

---